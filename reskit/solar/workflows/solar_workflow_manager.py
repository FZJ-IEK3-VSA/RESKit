import geokit as gk
import pandas as pd
import numpy as np

from os.path import isfile
from collections import OrderedDict
from types import FunctionType
import warnings
from scipy.interpolate import RectBivariateSpline
import json

# from reskit import solarpower

from .. import core as rk_solar_core
from ...workflow_manager import WorkflowManager

# Lazily import PVLib
import importlib

"""

Importing required packages.

"""


class LazyLoader:
    def __init__(self, lib_name):
        """
        LazyLoader is a utility class which postpones the "real" importing of the desired module until the time when it is actually needed
        """

        self.lib_name = lib_name
        self._mod = None

    def __getattr__(self, name):
        if self._mod is None:
            self._mod = importlib.import_module(self.lib_name)
        return getattr(self._mod, name)


pvlib = LazyLoader("pvlib")


class SolarWorkflowManager(WorkflowManager):
    def __init__(self, placements):
        """

        __init_(self, placements)

        Initialization of an instance of the generic SolarWorkflowManager class.

        Parameters
        ----------
        placements : pandas Dataframe
                     The locations that the simulation should be run for.
                     Columns must include "lon", "lat"

        Returns
        -------
        SolarWorkflorManager

        """

        # Do basic workflow construction
        super().__init__(placements)
        self._time_sel_ = None
        self._time_index_ = None
        self.module = None

    def estimate_tilt_from_latitude(self, convention):
        """

        estimate_tilt_from_latitude(self, convention)

        Estimates the tilt of the solar panels based on the latitude of the placements of the instance.

        Parameters
        ----------
        convention : str, optional
                     The calculation method used to suggest system tilts.
                     Option 1 of convention is "Ryberg2020".
                     Option 2 of convention is a string consumable by 'eval'. This string can use the variable latitude.
                     For example "latitude*0.76".
                     Option 3 of convention is a path to a rasterfile.
                     To get more information check out reskit.solar.location_to_tilt for more information.



        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object

        """

        self.placements["tilt"] = rk_solar_core.system_design.location_to_tilt(
            self.locs, convention=convention
        )
        return self

    def estimate_azimuth_from_latitude(self):
        """

        estimate_azimuth_from_latitude(self)

        Estimates the azimuth of the placements of the instance.
        For a positive latitude the azimuth is set to 180.
        For a negative latitude the azimuth is set to 0.

        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object

        """

        self.placements["azimuth"] = 180

        self.placements["azimuth"].values[self.locs.lats < 0] = 0
        return self

    def apply_elevation(self, elev):
        """

        apply_elevation(self)

        Adds an elevation (name: 'elev') column to the placements data frame.

        Parameters
        ----------
        elev: str, list
              If a string is given it must be a path to a rasterfile including the elevations.
              If a list is given it has to include the elevations at each location.


        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object

        """

        if isinstance(elev, str):
            clipped_elev = self.ext.pad(0.5).rasterMosaic(elev)
            self.placements["elev"] = gk.raster.interpolateValues(
                clipped_elev, self.locs
            )
        else:
            self.placements["elev"] = elev

        return self

    def determine_solar_position(
        self, lon_rounding=1, lat_rounding=1, elev_rounding=-2
    ):
        """

        determine_solar_position(self, lon_rounding=1, lat_rounding=1, elev_rounding=-2)

        Calculates azimuth and apparent zenith for each location using the pvlib fuction pvlib.solarposition.spa_python() [1].
        Adds azimuth and apparent zenit to the sim_data dictionary.


        Parameters
        ----------
        lon_rounding: int, optional
                      Decimal places that the longitude should be rounded to. Default is 1.

        lat_rounding: int, optional
                      Decimal places that the latitude should be rounded to. Default is 1.

        elev_rounding: int, optional
                      Decimal places that the elevation should be rounded to. Default is -2.

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object

        Notes
        -----
        Required columns in the placements dataframe to use this functions are 'lon', 'lat' and 'elev'.
        Required data in the sim_data dictionary are 'surface_pressure' and 'surface_air_temperature'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.solarposition.spa_python.html

        [2] I. Reda and A. Andreas, Solar position algorithm for solar
            radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.

        [3] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
            solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838,
            2007.

        [4] USNO delta T:
            http://www.usno.navy.mil/USNO/earth-orientation/eo-products/long-term


        """

        assert "lon" in self.placements.columns
        assert "lat" in self.placements.columns
        assert "elev" in self.placements.columns
        assert "surface_pressure" in self.sim_data
        assert "surface_air_temperature" in self.sim_data

        rounded_locs = pd.DataFrame()
        rounded_locs["lon"] = np.round(self.placements["lon"].values, lon_rounding)
        rounded_locs["lat"] = np.round(self.placements["lat"].values, lat_rounding)
        rounded_locs["elev"] = np.round(self.placements["elev"].values, elev_rounding)

        solar_position_library = dict()

        # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)
        self.sim_data["solar_azimuth"] = np.full_like(
            self.sim_data["surface_pressure"], np.nan
        )
        # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)
        self.sim_data["apparent_solar_zenith"] = np.full_like(
            self.sim_data["surface_pressure"], np.nan
        )
        # self.sim_data['apparent_solar_elevation'] = np.full_like(self.sim_data['surface_pressure'], np.nan)  # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)

        for loc, row in enumerate(rounded_locs.itertuples()):
            key = (row.lon, row.lat, row.elev)
            if key in solar_position_library:
                _solpos_ = solar_position_library[key]
            else:
                _solpos_ = pvlib.solarposition.spa_python(
                    self.time_index,
                    latitude=row.lat,
                    longitude=row.lon,
                    altitude=row.elev,
                    pressure=self.sim_data["surface_pressure"][:, loc],
                    temperature=self.sim_data["surface_air_temperature"][:, loc],
                )
                solar_position_library[key] = _solpos_

            self.sim_data["solar_azimuth"][:, loc] = _solpos_["azimuth"]
            self.sim_data["apparent_solar_zenith"][:, loc] = _solpos_["apparent_zenith"]
            # self.sim_data['apparent_solar_elevation'][:, loc] = _solpos_["apparent_elevation"]

        assert not np.isnan(self.sim_data["solar_azimuth"]).any()
        assert not np.isnan(self.sim_data["apparent_solar_zenith"]).any()
        # assert not np.isnan(self.sim_data['apparent_solar_elevation']).any()

        return self

    def filter_positive_solar_elevation(self):
        """

        filter_positive_solar_elevation(self)

        Filters positive solar elevations so that future operations are only executed for time steps when the sun is above (or at least near-to) the horizon


        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object

        Notes
        -----
        Required data in the sim_data dictionary are 'apparent_solar_zenith'.



        """

        if self._time_sel_ is not None:
            warnings.warn("Filtering already applied, skipping...")
            return self
        assert "apparent_solar_zenith" in self.sim_data

        self._time_sel_ = (self.sim_data["apparent_solar_zenith"] < 95).any(axis=1)

        for key in self.sim_data.keys():
            self.sim_data[key] = self.sim_data[key][self._time_sel_, :]

        self._time_index_ = self.time_index[self._time_sel_]
        self._set_sim_shape()

        return self

    def determine_extra_terrestrial_irradiance(self, **kwargs):
        """

        determine_extra_terrestrial_irradiance(self, **kwargs)

        Determines extra terrestrial irradiance using the pvlib.irradiance.get_extra_radiation() function [1].

        Parameters
        ----------
        None

        Returns
        -------

        Returns a reference to the invoking SolarWorkflowManager object.


        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.irradiance.get_extra_radiation.html

        [2]	M. Reno, C. Hansen, and J. Stein, “Global Horizontal Irradiance Clear Sky Models: Implementation and Analysis”, Sandia National Laboratories, SAND2012-2389, 2012.

        [3]	<http://solardat.uoregon.edu/SolarRadiationBasics.html>, Eqs. SR1 and SR2

        [4]	Partridge, G. W. and Platt, C. M. R. 1976. Radiative Processes in Meteorology and Climatology.

        [5]	Duffie, J. A. and Beckman, W. A. 1991. Solar Engineering of Thermal Processes, 2nd edn. J. Wiley and Sons, New York.

        [6]	ASCE, 2005. The ASCE Standardized Reference Evapotranspiration Equation, Environmental and Water Resources Institute of the American Civil Engineers, Ed. R. G. Allen et al.

        """

        dni_extra = pvlib.irradiance.get_extra_radiation(
            self._time_index_, **kwargs
        ).values

        shape = len(self._time_index_), self.locs.count
        self.sim_data["extra_terrestrial_irradiance"] = np.broadcast_to(
            dni_extra.reshape((shape[0], 1)), shape
        )

        return self

    def determine_air_mass(self, model="kastenyoung1989"):
        """

        determine_air_mass(self, model='kastenyoung1989')

        Determines air mass using the pvlib function pvlib.atmosphere.get_relative_airmass() [1].


        Parameters
        ----------
        model: str, optional
               default 'kastenyoung1989' [1]

               'simple' - secant(apparent zenith angle) - Note that this gives -inf at zenith=90 [2]
               'kasten1966' - See reference [2] - requires apparent sun zenith [2]
               'youngirvine1967' - See reference [3] - requires true sun zenith [2]
               'kastenyoung1989' - See reference [4] - requires apparent sun zenith [2]
               'gueymard1993' - See reference [5] - requires apparent sun zenith [2]
               'young1994' - See reference [6] - requries true sun zenith [2]
               'pickering2002' - See reference [7] - requires apparent sun zenith [2]



        Returns
        -------
        Nothing is returned.

        Notes
        -----
        Required data in the sim_data dictionary are 'apparent_solar_zenith'.



        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.atmosphere.get_relative_airmass.html

        [2]	Fritz Kasten. “A New Table and Approximation Formula for the Relative Optical Air Mass”. Technical Report 136, Hanover, N.H.: U.S. Army Material Command, CRREL.

        [3]	A. T. Young and W. M. Irvine, “Multicolor Photoelectric Photometry of the Brighter Planets,” The Astronomical Journal, vol. 72, pp. 945-950, 1967.

        [4]	Fritz Kasten and Andrew Young. “Revised optical air mass tables and approximation formula”. Applied Optics 28:4735-4738

        [5]	C. Gueymard, “Critical analysis and performance assessment of clear sky solar irradiance models using theoretical and measured data,” Solar Energy, vol. 51, pp. 121-138, 1993.

        [6]	A. T. Young, “AIR-MASS AND REFRACTION,” Applied Optics, vol. 33, pp. 1108-1110, Feb 1994.

        [7]	Keith A. Pickering. “The Ancient Star Catalog”. DIO 12:1, 20,

        [8]	Matthew J. Reno, Clifford W. Hansen and Joshua S. Stein, “Global Horizontal Irradiance Clear Sky Models: Implementation and Analysis” Sandia Report, (2012).


        """

        assert "apparent_solar_zenith" in self.sim_data

        # 29 becasue that what the function seems to max out at as zenith approaches 90
        self.sim_data["air_mass"] = np.full_like(
            self.sim_data["apparent_solar_zenith"], 29
        )

        s = self.sim_data["apparent_solar_zenith"] < 90
        self.sim_data["air_mass"][s] = pvlib.atmosphere.get_relative_airmass(
            self.sim_data["apparent_solar_zenith"][s], model=model
        )

    def apply_DIRINT_model(self, use_pressure=False, use_dew_temperature=False):
        """

        apply_DIRINT_model(self, use_pressure=False, use_dew_temperature=False)

        Determines direct normal irradiance (DNI) using the pvlib.irradiance.dirint() function [1].


        Parameters
        ----------
        use_pressure: boolian, optional
                      Default: False

        use_dew_temperature: boolian, optional
                             Default: False

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'global_horizontal_irradiance', 'surface_pressure',
        'surface_dew_temperature', 'apparent_solar_zenith', 'air_mass' and 'extra_terrestrial_irradiance'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.irradiance.dirint.html

        [2]	Perez, R., P. Ineichen, E. Maxwell, R. Seals and A. Zelenka, (1992). “Dynamic Global-to-Direct Irradiance Conversion Models”. ASHRAE Transactions-Research Series, pp. 354-369

        [3]	Maxwell, E. L., “A Quasi-Physical Model for Converting Hourly Global Horizontal to Direct Normal Insolation”, Technical Report No. SERI/TR-215-3087, Golden, CO: Solar Energy Research Institute, 1987.


        """

        assert "global_horizontal_irradiance" in self.sim_data
        assert "surface_pressure" in self.sim_data
        assert "surface_dew_temperature" in self.sim_data
        assert "apparent_solar_zenith" in self.sim_data
        assert "air_mass" in self.sim_data
        assert "extra_terrestrial_irradiance" in self.sim_data

        # self.sim_data["direct_normal_irradiance"] = solarpower.myDirint(
        #     ghi=self.sim_data['global_horizontal_irradiance'],
        #     zenith=self.sim_data["apparent_solar_zenith"],
        #     pressure=self.sim_data["surface_pressure"],
        #     amRel=self.sim_data["air_mass"],
        #     I0=self.sim_data["extra_terrestrial_irradiance"],
        #     temp_dew=self.sim_data["surface_dew_temperature"],
        #     use_delta_kt_prime=True,)

        use_pressure = True
        use_dew_temperature = True

        g = self.sim_data["global_horizontal_irradiance"].flatten()
        z = self.sim_data["apparent_solar_zenith"].flatten()
        p = self.sim_data["surface_pressure"].flatten() if use_pressure else None
        td = (
            self.sim_data["surface_dew_temperature"].flatten()
            if use_dew_temperature
            else None
        )
        times = pd.DatetimeIndex(
            np.column_stack(
                [self._time_index_ for x in range(self._sim_shape_[1])]
            ).flatten()
        )

        self.sim_data["direct_normal_irradiance"] = (
            pvlib.irradiance.dirint(
                ghi=g, solar_zenith=z, times=times, pressure=p, temp_dew=td
            )
            .fillna(0)
            .values.reshape(self._sim_shape_)
        )

        return self

    def diffuse_horizontal_irradiance_from_trigonometry(self):
        """

        diffuse_horizontal_irradiance_from_trigonometry(self)

        Calculates the diffuse horizontal irradiance from global horizontal irradiance, direct normal irradiance and apparent zenith.

        [TODO: Add a simple equation such as the one given in 'direct_normal_irradiance_from_trigonometry']

        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'global_horizontal_irradiance', 'direct_normal_irradiance' and
        'apparent_solar_zenith'.

        """

        assert "global_horizontal_irradiance" in self.sim_data
        assert "direct_normal_irradiance" in self.sim_data
        assert "apparent_solar_zenith" in self.sim_data

        ghi = self.sim_data["global_horizontal_irradiance"]
        dni = self.sim_data["direct_normal_irradiance"]
        elev = np.radians(90 - self.sim_data["apparent_solar_zenith"])

        self.sim_data["diffuse_horizontal_irradiance"] = ghi - dni * np.sin(elev)
        self.sim_data["diffuse_horizontal_irradiance"][
            self.sim_data["diffuse_horizontal_irradiance"] < 0
        ] = 0

        return self

    def direct_normal_irradiance_from_trigonometry(self):
        """

        direct_normal_irradiance_from_trigonometry(self):


        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required columns in the placements dataframe to use this functions are 'lon', 'lat' and 'elev'.
        Required data in the sim_data dictionary are 'direct_horizontal_irradiance' and 'apparent_solar_zenith'.

        Calculates the direct normal irradiance from the following equation:
            .. math:: dir_nor_irr = dir_hor_irr / cos( solar_zenith )

            Where:
            dir_nor_irr  -> The direct irradiance on the normal plane
            dir_hor_irr  -> The direct irradiance on the horizontal plane
            solar_zenith -> The solar zenith angle in radians

        """

        # TODO: This can also cover the case when we know GHI & DiffHI
        assert "direct_horizontal_irradiance" in self.sim_data
        assert "apparent_solar_zenith" in self.sim_data

        dni_flat = self.sim_data["direct_horizontal_irradiance"]
        zen = np.radians(self.sim_data["apparent_solar_zenith"])

        self.sim_data["direct_normal_irradiance"] = dni_flat / np.maximum(
            np.cos(zen), 0.2
        )

        # catch outliners from zero devision
        index_out = (dni_flat < 25) & (np.cos(zen) < 0.05)
        self.sim_data["direct_normal_irradiance"][index_out] = 0

        index_out = (dni_flat < 25) & (np.cos(zen) < 0.05)
        self.sim_data["direct_normal_irradiance"][index_out] = 0

        sel = ~np.isfinite(self.sim_data["direct_normal_irradiance"])
        sel = np.logical_or(sel, self.sim_data["direct_normal_irradiance"] < 0)
        sel = np.logical_or(sel, self.sim_data["direct_normal_irradiance"] > 1600)

        self.sim_data["direct_normal_irradiance"][sel] = 0

        return self

    def permit_single_axis_tracking(self, max_angle=90, backtrack=True, gcr=2.0 / 7.0):
        """

        permit_single_axis_tracking(self, max_angle=90, backtrack=True, gcr=2.0 / 7.0)

        Permits single axis tracking in the simulation using the pvlib.tracking.singleaxis() function [1].


        Parameters
        ----------
        max_angle: float, optional
                   default 90
                   A value denoting the maximum rotation angle, in decimal degrees, of the one-axis tracker from its horizontal position
                   (horizontal if axis_tilt = 0). A max_angle of 90 degrees allows the tracker to rotate to a vertical position to point the
                   panel towards a horizon. max_angle of 180 degrees allows for full rotation [1].

        backtrack: bool, optional
                   default True
                   Controls whether the tracker has the capability to “backtrack” to avoid row-to-row shading.
                   False denotes no backtrack capability. True denotes backtrack capability [1].

        gcr:       float, optional
                   default 2.0/7.0
                   A value denoting the ground coverage ratio of a tracker system which utilizes backtracking; i.e. the ratio between the
                   PV array surface area to total ground area. A tracker system with modules 2 meters wide, centered on the tracking axis,
                   with 6 meters between the tracking axes has a gcr of 2/6=0.333. If gcr is not provided, a gcr of 2/7 is default. gcr must be <=1 [1].


        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required columns in the placements dataframe to use this functions are 'lon', 'lat', 'elev', 'tilt' and 'azimuth'.
        Required data in the sim_data dictionary are 'apparent_solar_zenith' and 'solar_azimuth'.

        References
        ----------
        [1] https://wholmgren-pvlib-python-new.readthedocs.io/en/doc-reorg2/generated/tracking/pvlib.tracking.singleaxis.html

        [2]	Lorenzo, E et al., 2011, “Tracking and back-tracking”, Prog. in Photovoltaics: Research and Applications, v. 19, pp. 747-753.

        """

        """See pvlib.tracking.singleaxis for parameter info"""
        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data
        assert "tilt" in self.placements.columns
        assert "azimuth" in self.placements.columns

        self.register_workflow_parameter("tracking_mode", "single_axis")
        self.register_workflow_parameter("tracking_max_angle", max_angle)
        self.register_workflow_parameter("tracking_backtrack", backtrack)
        self.register_workflow_parameter("tracking_gcr", gcr)

        system_tilt = np.empty(self._sim_shape_)
        system_azimuth = np.empty(self._sim_shape_)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(self.locs.count):
                placement = self.placements.iloc[i]

                tmp = pvlib.tracking.singleaxis(
                    apparent_zenith=pd.Series(
                        self.sim_data["apparent_solar_zenith"][:, i],
                        index=self._time_index_,
                    ),
                    apparent_azimuth=pd.Series(
                        self.sim_data["solar_azimuth"][:, i], index=self._time_index_
                    ),
                    # self.placements['tilt'].values,
                    axis_tilt=placement.tilt,
                    # self.placements['azimuth'].values,
                    axis_azimuth=placement.azimuth,
                    max_angle=max_angle,
                    backtrack=backtrack,
                    gcr=gcr,
                )

                system_tilt[:, i] = tmp["surface_tilt"].values
                system_azimuth[:, i] = tmp["surface_azimuth"].values

                # fix nan values. Why are they there???
                s = np.isnan(system_tilt[:, i])
                system_tilt[s, i] = placement.tilt

                s = np.isnan(system_azimuth[:, i])
                system_azimuth[s, i] = placement.azimuth

        self.sim_data["system_tilt"] = system_tilt
        self.sim_data["system_azimuth"] = system_azimuth

        return self

    def determine_angle_of_incidence(self):
        """

        determine_angle_of_incidence(self)

        Determines the angle of incidence [TODO: credit the PVLib function as you've done in previous examples].

        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'apparent_solar_zenith' and 'solar_azimuth'.

        """

        """tracking can be: 'fixed' or 'singleaxis'"""
        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data

        azimuth = self.sim_data.get("system_azimuth", self.placements["azimuth"].values)
        tilt = self.sim_data.get("system_tilt", self.placements["tilt"].values)

        self.sim_data["angle_of_incidence"] = np.nan_to_num(
            pvlib.irradiance.aoi(
                tilt,
                azimuth,
                self.sim_data["apparent_solar_zenith"],
                self.sim_data["solar_azimuth"],
            ),
            0,
        )

        return self

    def estimate_plane_of_array_irradiances(
        self, transposition_model="perez", albedo=0.25, **kwargs
    ):
        """
        estimate_plane_of_array_irradiances(self, transposition_model="perez", albedo=0.25, **kwargs)

        Estimates the plane of array irradiance using the pvlib.irradiance.get_total_irradiance() function [1].


        Parameters
        ----------
        transportion_model: str, optional
                            default "perez"

        albedo: numeric, optional
                default 0.25
                Surface albedo [1].

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'apparent_solar_zenith', 'solar_azimuth', 'direct_normal_irradiance',
        'global_horizontal_irradiance', 'diffuse_horizontal_irradiance', 'extra_terrestrial_irradiance' and 'air_mass'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.irradiance.get_total_irradiance.html

        """

        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data
        assert "direct_normal_irradiance" in self.sim_data
        assert "global_horizontal_irradiance" in self.sim_data
        assert "diffuse_horizontal_irradiance" in self.sim_data
        assert "extra_terrestrial_irradiance" in self.sim_data
        assert "air_mass" in self.sim_data

        azimuth = self.sim_data.get("system_azimuth", self.placements["azimuth"].values)
        tilt = self.sim_data.get("system_tilt", self.placements["tilt"].values)

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=self.sim_data["apparent_solar_zenith"],
            solar_azimuth=self.sim_data["solar_azimuth"],
            dni=self.sim_data["direct_normal_irradiance"],
            ghi=self.sim_data["global_horizontal_irradiance"],
            dhi=self.sim_data["diffuse_horizontal_irradiance"],
            dni_extra=self.sim_data["extra_terrestrial_irradiance"],
            airmass=self.sim_data["air_mass"],
            albedo=albedo,
            model=transposition_model,
            **kwargs,
        )

        for key in poa.keys():
            # This should set: 'poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse', and 'poa_ground_diffuse'

            tmp = poa[key]
            tmp[np.isnan(tmp)] = 0

            self.sim_data[key] = tmp

        self._fix_bad_plane_of_array_values()

        return self

    def _fix_bad_plane_of_array_values(self):
        bad_poa = self.sim_data["poa_global"] >= 1600
        if (bad_poa).any():
            # POA is super big, but this only happens when elevation angles are approximately
            # zero (sin effect), so it should be okay to just set the POA to zero as well
            self.sim_data["poa_global"] = np.where(
                bad_poa, 0, self.sim_data["poa_global"]
            )
            self.sim_data["poa_direct"] = np.where(
                bad_poa, 0, self.sim_data["poa_direct"]
            )
            self.sim_data["poa_diffuse"] = np.where(
                bad_poa, 0, self.sim_data["poa_diffuse"]
            )
            self.sim_data["poa_sky_diffuse"] = np.where(
                bad_poa, 0, self.sim_data["poa_sky_diffuse"]
            )
            self.sim_data["poa_ground_diffuse"] = np.where(
                bad_poa, 0, self.sim_data["poa_ground_diffuse"]
            )

    def cell_temperature_from_sapm(self, mounting="glass_open_rack"):
        """
        cell_temperature_from_sapm(self, mounting="glass_open_rack")

        Calculates the cell temperature based on the pvlib.temperature.sapm_cell() function [1].


        Parameters
        ----------
        mounting: str
                  Options:
                  "glass_open_rack" [1]
                  "glass_close_roof" [1]
                  "polymer_open_rack" [1]
                  "polymer_insulated_back" [1]

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'surface_wind_speed', 'surface_air_temperature' and 'poa_global'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.temperature.sapm_cell.html


        """
        assert "surface_wind_speed" in self.sim_data
        assert "surface_air_temperature" in self.sim_data
        assert "poa_global" in self.sim_data

        if mounting == "glass_open_rack":
            a, b, dT = -3.47, -0.0594, 3
        elif mounting == "glass_close_roof":
            a, b, dT = -2.98, -0.0471, 1
        elif mounting == "polymer_open_rack":
            a, b, dT = -3.56, -0.075, 3
        elif mounting == "polymer_insulated_back":
            a, b, dT = -2.81, -0.0455, 0
        else:
            raise RuntimeError(
                "mounting not one of: 'glass_open_rack', 'glass_close_roof', 'polymer_open_rack', or 'polymer_insulated_back'"
            )

        self.sim_data["cell_temperature"] = pvlib.temperature.sapm_cell(
            self.sim_data["poa_global"],
            self.sim_data["surface_air_temperature"],
            self.sim_data["surface_wind_speed"],
            a=a,
            b=b,
            deltaT=dT,
            irrad_ref=1000,
        )

        return self

    def apply_angle_of_incidence_losses_to_poa(self):
        """
        apply_angle_of_incidence_losses_to_poa(self)


        Applies the angle of incidence losses to the plane-of-array irradiance using the pvlib.pvsystem.iam.physical() function [1].

        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required data in the sim_data dictionary are 'poa_direct', 'poa_ground_diffuse' and 'poa_sky_diffuse'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.iam.physical.html


        """

        assert "poa_direct" in self.sim_data
        assert "poa_ground_diffuse" in self.sim_data
        assert "poa_sky_diffuse" in self.sim_data

        tilt = self.sim_data.get("system_tilt", self.placements["tilt"].values)

        self.sim_data["poa_direct"] *= pvlib.pvsystem.iam.physical(
            aoi=self.sim_data["angle_of_incidence"],
            n=1.526,  # PVLIB v0.7.2 default
            K=4.0,  # PVLIB v0.7.2 default
            L=0.002,  # PVLIB v0.7.2 default
        )

        # Effective angle of incidence values from "Solar-Engineering-of-Thermal-Processes-4th-Edition"
        self.sim_data["poa_ground_diffuse"] *= pvlib.pvsystem.iam.physical(
            aoi=(90 - 0.5788 * tilt + 0.002693 * np.power(tilt, 2)),
            n=1.526,  # PVLIB v0.7.2 default
            K=4.0,  # PVLIB v0.7.2 default
            L=0.002,  # PVLIB v0.7.2 default
        )

        self.sim_data["poa_sky_diffuse"] *= pvlib.pvsystem.iam.physical(
            aoi=(59.7 - 0.1388 * tilt + 0.001497 * np.power(tilt, 2)),
            n=1.526,  # PVLIB v0.7.2 default
            K=4.0,  # PVLIB v0.7.2 default
            L=0.002,  # PVLIB v0.7.2 default
        )

        self.sim_data["poa_diffuse"] = (
            self.sim_data["poa_ground_diffuse"] + self.sim_data["poa_sky_diffuse"]
        )
        self.sim_data["poa_global"] = (
            self.sim_data["poa_direct"] + self.sim_data["poa_diffuse"]
        )

        assert (self.sim_data["poa_global"] < 1600).all(), "POA is too large"

        return self

    def configure_cec_module(self, module="WINAICO WSx-240P6"):
        """
        configure_cec_module(self, module="WINAICO WSx-240P6")

        Configures CEC of a module based on the outputs of the pvlib.pvsystem.retrieve_sam() function [1].

        Parameters
        ----------
        module: str or dict
            Must be one of:
                * A module found in the pvlib.pvsystem.retrieve_sam("CECMod") database
                * "WINAICO WSx-240P6" -> Good for open-field applications
                * "LG Electronics LG370Q1C-A5" -> Good for rooftop applications
                * A dict containing a set of module parameters, including:
                    T_NOCT, A_c, N_s, I_sc_ref, V_oc_ref, I_mp_ref, V_mp_ref, alpha_sc,
                    beta_oc, a_ref, I_L_ref, I_o_ref, R_s, R_sh_ref, Adjust, gamma_r, PTC

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.retrieve_sam.html


        """

        if isinstance(module, str):
            self.register_workflow_parameter("module_name", module)

            if module == "WINAICO WSx-240P6":
                module = pd.Series(
                    dict(
                        BIPV="N",
                        Date="6/2/2014",
                        T_NOCT=43,
                        A_c=1.663,
                        N_s=60,
                        I_sc_ref=8.41,
                        V_oc_ref=37.12,
                        I_mp_ref=7.96,
                        V_mp_ref=30.2,
                        alpha_sc=0.001164,
                        beta_oc=-0.12357,
                        a_ref=1.6704,
                        I_L_ref=8.961,
                        I_o_ref=1.66e-11,
                        R_s=0.405,
                        R_sh_ref=326.74,
                        Adjust=4.747,
                        gamma_r=-0.383,
                        Version="NRELv1",
                        PTC=220.2,
                        Technology="Multi-c-Si",
                    )
                )
                module.name = "WINAICO WSx-240P6"
            elif module == "LG Electronics LG370Q1C-A5":
                module = pd.Series(
                    dict(
                        BIPV="N",
                        Date="12/14/2016",
                        T_NOCT=45.7,
                        A_c=1.673,
                        N_s=60,
                        I_sc_ref=10.82,
                        V_oc_ref=42.8,
                        I_mp_ref=10.01,
                        V_mp_ref=37,
                        alpha_sc=0.003246,
                        beta_oc=-0.10272,
                        a_ref=1.5532,
                        I_L_ref=10.829,
                        I_o_ref=1.12e-11,
                        R_s=0.079,
                        R_sh_ref=92.96,
                        Adjust=14,
                        gamma_r=-0.32,
                        Version="NRELv1",
                        PTC=347.2,
                        Technology="Mono-c-Si",
                    )
                )
                module.name = "LG Electronics LG370Q1C-A5"
            elif isinstance(module, str):
                # Extract module parameters
                db = pvlib.pvsystem.retrieve_sam("CECMod")
                try:
                    module = getattr(db, module)
                except:
                    raise RuntimeError(
                        "The module '{}' is not in the CEC database".format(module)
                    )
        else:
            module = pd.Series(module)
            assert "T_NOCT" in module.index
            assert "A_c" in module.index
            assert "N_s" in module.index
            assert "I_sc_ref" in module.index
            assert "V_oc_ref" in module.index
            assert "I_mp_ref" in module.index
            assert "V_mp_ref" in module.index
            assert "alpha_sc" in module.index
            assert "beta_oc" in module.index
            assert "a_ref" in module.index
            assert "I_L_ref" in module.index
            assert "I_o_ref" in module.index
            assert "R_s" in module.index
            assert "R_sh_ref" in module.index
            assert "Adjust" in module.index
            assert "gamma_r" in module.index
            assert "PTC" in module.index

            try:
                module_desc = json.dumps(module)
            except:
                module_desc = "user-configured"
            self.register_workflow_parameter("module_desc", module_desc)

        # # Check if we need to add the Desoto parameters
        # # defaults for EgRef and dEgdT taken from the note in the docstring for
        # #  'pvlib.pvsystem.calcparams_desoto'
        # if not "EgRef" in module:
        #     module['EgRef'] = 1.121
        # if not "dEgdT" in module:
        #     module['dEgdT'] = -0.0002677

        self.module = module

        return self

    def simulate_with_interpolated_single_diode_approximation(
        self, module="WINAICO WSx-240P6"
    ):
        """
        simulate_with_interpolated_single_diode_approximation(self, module="WINAICO WSx-240P6")

        Does the simulation with an interpolated single diode approximation using the pvlib.pvsystem.calcparams_desoto() [1] function and the
        pvlib.pvsystem.singlediode() [2] function.


        Parameters
        ----------
        module: str
            Must be one of:
                * A module found in the pvlib.pvsystem.retrieve_sam("CECMod") database
                * "WINAICO WSx-240P6" -> Good for open-field applications
                * "LG Electronics LG370Q1C-A5" -> Good for rooftop applications

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required columns in the placements dataframe to use this functions are 'lon', 'lat', 'elev', 'tilt' and 'azimuth'.
        Required data in the sim_data dictionary are 'poa_global' and 'cell_temperature'.

        References
        ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.calcparams_desoto.html

        [2] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.singlediode.html

        [3]	(1, 2) W. De Soto et al., “Improvement and validation of a model for photovoltaic array performance”, Solar Energy, vol 80, pp. 78-88, 2006.

        [4]	System Advisor Model web page. https://sam.nrel.gov.

        [5]	A. Dobos, “An Improved Coefficient Calculator for the California Energy Commission 6 Parameter Photovoltaic Module Model”, Journal of Solar Energy Engineering, vol 134, 2012.

        [6]	O. Madelung, “Semiconductors: Data Handbook, 3rd ed.” ISBN 3-540-40488-0

        [7]	S.R. Wenham, M.A. Green, M.E. Watt, “Applied Photovoltaics” ISBN 0 86758 909 4

        [8]	A. Jain, A. Kapoor, “Exact analytical solutions of the parameters of real solar cells using Lambert W-function”, Solar Energy Materials and Solar Cells, 81 (2004) 269-277.

        [9]	D. King et al, “Sandia Photovoltaic Array Performance Model”, SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

        [10]	“Computer simulation of the effects of electrical mismatches in photovoltaic cell interconnection circuits” JW Bishop, Solar Cell (1988) https://doi.org/10.1016/0379-6787(88)90059-2

        """

        """
        TODO: Make it work with multiple module definitions
        """
        assert "poa_global" in self.sim_data
        assert "cell_temperature" in self.sim_data

        self.configure_cec_module(module)

        sel = self.sim_data["poa_global"] > 0

        poa = self.sim_data["poa_global"][sel]
        cell_temp = self.sim_data["cell_temperature"][sel]

        # Use RectBivariateSpline to speed up simulation, but at the cost of accuracy (should still be >99.996%)
        maxpoa = np.nanmax(poa)

        _poa = np.concatenate(
            [
                np.logspace(-1, np.log10(maxpoa / 10), 20, endpoint=False),
                np.linspace(maxpoa / 10, maxpoa, 80),
            ]
        )
        _temp = np.linspace(cell_temp.min(), cell_temp.max(), 100)
        poaM, tempM = np.meshgrid(_poa, _temp)

        sotoParams = pvlib.pvsystem.calcparams_desoto(
            effective_irradiance=poaM.flatten(),
            temp_cell=tempM.flatten(),
            alpha_sc=self.module.alpha_sc,
            a_ref=self.module.a_ref,
            I_L_ref=self.module.I_L_ref,
            I_o_ref=self.module.I_o_ref,
            R_sh_ref=self.module.R_sh_ref,
            R_s=self.module.R_s,
            EgRef=1.121,  # PVLIB v0.7.2 Default
            dEgdT=-0.0002677,  # PVLIB v0.7.2 Default
            irrad_ref=1000,  # PVLIB v0.7.2 Default
            temp_ref=25,  # PVLIB v0.7.2 Default
        )

        photoCur, satCur, resSeries, resShunt, nNsVth = sotoParams
        gen = pvlib.pvsystem.singlediode(
            photocurrent=photoCur,
            saturation_current=satCur,
            resistance_series=resSeries,
            resistance_shunt=resShunt,
            nNsVth=nNsVth,
            ivcurve_pnts=None,  # PVLIB v0.7.2 Default
            method="lambertw",  # PVLIB v0.7.2 Default
        )

        interpolator = RectBivariateSpline(
            _temp, _poa, gen["p_mp"].reshape(poaM.shape), kx=3, ky=3
        )
        self.sim_data["module_dc_power_at_mpp"] = np.zeros_like(
            self.sim_data["poa_global"]
        )
        self.sim_data["module_dc_power_at_mpp"][sel] = interpolator(
            cell_temp, poa, grid=False
        )

        interpolator = RectBivariateSpline(
            _temp, _poa, gen["v_mp"].reshape(poaM.shape), kx=3, ky=3
        )
        self.sim_data["module_dc_voltage_at_mpp"] = np.zeros_like(
            self.sim_data["poa_global"]
        )
        self.sim_data["module_dc_voltage_at_mpp"][sel] = interpolator(
            cell_temp, poa, grid=False
        )

        self.sim_data["capacity_factor"] = self.sim_data["module_dc_power_at_mpp"] / (
            self.module.I_mp_ref * self.module.V_mp_ref
        )

        # Estimate total system generation
        if "capacity" in self.placements.columns:
            self.sim_data["total_system_generation"] = self.sim_data[
                "capacity_factor"
            ] * np.broadcast_to(self.placements.capacity, self._sim_shape_)

        if (
            "modules_per_string" in self.placements.columns
            and "strings_per_inverter" in self.placements.columns
        ):
            total_modules = (
                self.placements.modules_per_string
                * self.placements.strings_per_inverter
                * getattr(self.placements, "number_of_inverters", 1)
            )

            self.sim_data["total_system_generation"] = self.sim_data[
                "module_dc_power_at_mpp"
            ] * np.broadcast_to(total_modules, self._sim_shape_)

        return self

    def apply_inverter_losses(
        self,
        inverter,
        method="sandia",
    ):
        """
         apply_inverter_losses(self, inverter, method="sandia", )

         Applies inverter losses using the pvlib.pvsystem.snlinverter() fuction [1], the pvlib.pvsystem.retrieve_sam() fuction [2] and the
         pvlib.pvsystem.adrinverter() fuction [3].


         Parameters
         ----------
         inverter: str
                   Describes the inverter.
                   [TODO: Add a more detailed desciption following the example of 'configure_cec_module']
         method: str
                 Options:
                 "scandia"
                 "driesse"
                 Describes the used method to apply the inverter losses.

         Returns
         -------
         Returns a reference to the invoking SolarWorkflowManager object.

         Notes
         -----
         Required data in the sim_data dictionary are 'module_dc_power_at_mpp' and 'module_dc_voltage_at_mpp'.
         Required data in the placements dataframe are 'modules_per_string' and 'strings_per_inverter'.
         Cannot simultaneously provide 'capacity' and inverter-string parameters.


         References
         ----------
        [1] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.snlinverter.html

        [2] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.retrieve_sam.html

        [3] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.adrinverter.html

        [4]	SAND2007-5036, “Performance Model for Grid-Connected Photovoltaic Inverters by D. King, S. Gonzalez, G. Galbraith, W. Boyson

        [5]	System Advisor Model web page. https://sam.nrel.gov.

        [6]	Beyond the Curves: Modeling the Electrical Efficiency of Photovoltaic Inverters, PVSC 2008, Anton Driesse et. al.

        """

        """method can be: 'sandia' or 'driesse'

        TODO: Make it work with multiplt inverter definitions
        """

        assert "module_dc_power_at_mpp" in self.sim_data
        assert "module_dc_voltage_at_mpp" in self.sim_data
        assert hasattr(self, "module")
        assert "modules_per_string" in self.placements.columns
        assert "strings_per_inverter" in self.placements.columns
        assert (
            not "capacity" in self.placements.columns
        ), "Cannot simultaneously provide 'capacity' and inverter-string parameters"

        if method == "sandia":
            if isinstance(inverter, str):
                db = pvlib.pvsystem.retrieve_sam("SandiaInverter")
                inverter = getattr(db, inverter)

            self.sim_data["inverter_ac_power_at_mpp"] = pvlib.inverter.sandia(
                v_dc=self.sim_data["module_dc_voltage_at_mpp"]
                * np.broadcast_to(self.placements.modules_per_string, self._sim_shape_),
                p_dc=self.sim_data["module_dc_power_at_mpp"]
                * np.broadcast_to(
                    self.placements.modules_per_string
                    * self.placements.strings_per_inverter,
                    self._sim_shape_,
                ),
                inverter=inverter,
            )

        elif method == "driesse":
            if isinstance(inverter, str):
                db = pvlib.pvsystem.retrieve_sam("CECInverter")
                inverter = getattr(db, inverter)

            self.sim_data["inverter_ac_power_at_mpp"] = pvlib.pvsystem.adrinverter(
                v_dc=self.sim_data["module_dc_voltage_at_mpp"]
                * np.broadcast_to(self.placements.modules_per_string, self._sim_shape_),
                p_dc=self.sim_data["module_dc_power_at_mpp"]
                * np.broadcast_to(
                    self.placements.modules_per_string
                    * self.placements.strings_per_inverter,
                    self._sim_shape_,
                ),
                inverter=inverter,
            )

        number_of_inverters = getattr(self.placements, "number_of_inverters", 1)
        self.sim_data["total_system_generation"] = self.sim_data[
            "inverter_ac_power_at_mpp"
        ] * np.broadcast_to(number_of_inverters, self._sim_shape_)

        total_capacity = (
            self.module.I_mp_ref
            * self.module.V_mp_ref
            * self.placements.modules_per_string
            * self.placements.strings_per_inverter
            * number_of_inverters
        )

        self.sim_data["capacity_factor"] = self.sim_data[
            "total_system_generation"
        ] / np.broadcast_to(total_capacity, self._sim_shape_)

        return self

    # def to_xarray(self, output_netcdf_path=None):
    #     xds = super().to_xarray(_intermediate_dict=True)
