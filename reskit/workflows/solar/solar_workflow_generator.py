import geokit as gk
import pandas as pd
import numpy as np

from os.path import isfile
from collections import OrderedDict
from types import FunctionType
from warnings import warn
from scipy.interpolate import RectBivariateSpline

import reskit as rk
# from reskit import solarpower
from ..workflow_generator import WorkflowGenerator

# Lazily import PVLib
import importlib


class LazyLoader:
    def __init__(self, lib_name):
        self.lib_name = lib_name
        self._mod = None

    def __getattr__(self, name):
        if self._mod is None:
            self._mod = importlib.import_module(self.lib_name)
        return getattr(self._mod, name)


pvlib = LazyLoader("pvlib")


class SolarWorkflowGenerator(WorkflowGenerator):
    def __init__(self, placements):
        # Do basic workflow construction
        super().__init__(placements)
        self._time_sel_ = None
        self._time_index_ = None

    def estimate_tilt_from_latitude(self, convention):
        self.placements['tilt'] = solarpower.locToTilt(
            self.locs, convention=convention)
        return self

    def estimate_azimuth_from_latitude(self):
        self.placements['azimuth'] = 180
        self.placements['azimuth'][self.locs.lats < 0] = 0
        return self

    def apply_elevation(self, elev):
        if isinstance(elev, str):
            self.placements['elev'] = gk.raster.interpolateValues(
                elev, self.locs)
        else:
            self.placements['elev'] = elev

        return self

    def determine_solar_position(self, lon_rounding=1, lat_rounding=1, elev_rounding=-2):
        assert "lon" in self.placements.columns
        assert "lat" in self.placements.columns
        assert "elev" in self.placements.columns
        assert "surface_pressure" in self.sim_data
        assert "surface_air_temperature" in self.sim_data

        rounded_locs = pd.DataFrame()
        rounded_locs['lon'] = np.round(self.placements['lon'].values, lon_rounding)
        rounded_locs['lat'] = np.round(self.placements['lat'].values, lat_rounding)
        rounded_locs['elev'] = np.round(self.placements['elev'].values, elev_rounding)

        solar_position_library = dict()

        self.sim_data['solar_azimuth'] = np.full_like(self.sim_data['surface_pressure'], np.nan)  # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)
        self.sim_data['apparent_solar_zenith'] = np.full_like(self.sim_data['surface_pressure'], np.nan)  # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)
        self.sim_data['apparent_solar_elevation'] = np.full_like(self.sim_data['surface_pressure'], np.nan)  # pd.DataFrame(np.nan, index=self.time_index, columns=self.locs)

        for loc, row in enumerate(rounded_locs.itertuples()):
            key = (row.lon, row.lat, row.elev)
            if key in solar_position_library:
                _solpos_ = solar_position_library[key]
            else:
                _solpos_ = pvlib.solarposition.spa_python(self.time_index,
                                                          latitude=row.lat,
                                                          longitude=row.lon,
                                                          altitude=row.elev,
                                                          pressure=self.sim_data["surface_pressure"][:, loc],
                                                          temperature=self.sim_data["surface_air_temperature"][:, loc])
                solar_position_library[key] = _solpos_

            self.sim_data['solar_azimuth'][:, loc] = _solpos_["azimuth"]
            self.sim_data['apparent_solar_zenith'][:, loc] = _solpos_["apparent_zenith"]
            self.sim_data['apparent_solar_elevation'][:, loc] = _solpos_["apparent_elevation"]

        assert not np.isnan(self.sim_data['solar_azimuth']).any()
        assert not np.isnan(self.sim_data['apparent_solar_zenith']).any()
        # assert not np.isnan(self.sim_data['apparent_solar_elevation']).any()

        return self

    def filter_positive_solar_elevation(self):
        if self._time_sel_ is not None:
            warn("Filtering already applied, skipping...")
            return self
        assert "apparent_solar_zenith" in self.sim_data

        self._time_sel_ = (self.sim_data["apparent_solar_zenith"] < 92).any(axis=1)

        for key in self.sim_data.keys():
            self.sim_data[key] = self.sim_data[key][self._time_sel_, :]

        self._time_index_ = self.time_index[self._time_sel_]
        self._set_sim_shape()

        return self

    def determine_extra_terrestrial_irradiance(self, **kwargs):
        dni_extra = pvlib.irradiance.extraradiation(self._time_index_, **kwargs).values

        shape = len(self._time_index_), self.locs.count
        self.sim_data['extra_terrestrial_irradiance'] = np.broadcast_to(
            dni_extra.reshape((shape[0], 1)), shape)

        return self

    def determine_air_mass(self, model='kastenyoung1989'):
        assert "apparent_solar_zenith" in self.sim_data

        # 29 becasue that what the function seems to max out at as zenith approaches 90
        self.sim_data["air_mass"] = np.full_like(self.sim_data['apparent_solar_zenith'], 29)

        s = self.sim_data['apparent_solar_zenith'] < 90
        self.sim_data["air_mass"][s] = pvlib.atmosphere.relativeairmass(self.sim_data['apparent_solar_zenith'][s], model=model)

    def apply_DIRINT_model(self):
        assert "global_horizontal_irradiance" in self.sim_data
        assert "surface_pressure" in self.sim_data
        assert "surface_dew_temperature" in self.sim_data
        assert "apparent_solar_zenith" in self.sim_data
        assert "air_mass" in self.sim_data
        assert "extra_terrestrial_irradiance" in self.sim_data

        self.sim_data["direct_normal_irradiance"] = solarpower.myDirint(
            ghi=self.sim_data['global_horizontal_irradiance'],
            zenith=self.sim_data["apparent_solar_zenith"],
            pressure=self.sim_data["surface_pressure"],
            amRel=self.sim_data["air_mass"],
            I0=self.sim_data["extra_terrestrial_irradiance"],
            temp_dew=self.sim_data["surface_dew_temperature"],
            use_delta_kt_prime=True,)

        return self

    def diffuse_horizontal_irradiance_from_trigonometry(self):
        assert "global_horizontal_irradiance" in self.sim_data
        assert "direct_normal_irradiance" in self.sim_data
        assert "apparent_solar_zenith" in self.sim_data

        ghi = self.sim_data['global_horizontal_irradiance']
        dni = self.sim_data['direct_normal_irradiance']
        elev = np.radians(90 - self.sim_data['apparent_solar_zenith'])

        self.sim_data['diffuse_horizontal_irradiance'] = ghi - dni * np.sin(elev)
        self.sim_data['diffuse_horizontal_irradiance'][self.sim_data['diffuse_horizontal_irradiance'] < 0] = 0

        return self

    def permit_single_axis_tracking(self, max_angle=90, backtrack=True, gcr=2.0 / 7.0):
        """See pvlib.tracking.singleaxis for parameter info"""
        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data
        assert "tilt" in self.placements.columns
        assert "azimuth" in self.placements.columns

        self.register_workflow_parameter("tracking_mode", "fixed")
        self.register_workflow_parameter("tracking_max_angle", max_angle)
        self.register_workflow_parameter("tracking_backtrack", backtrack)
        self.register_workflow_parameter("tracking_gcr", gcr)

        system_tilt = np.empty(self._sim_shape_)
        system_azimuth = np.empty(self._sim_shape_)

        for i in range(self.locs.count):
            placement = self.placements.iloc[i]

            tmp = pvlib.tracking.singleaxis(
                apparent_zenith=pd.Series(self.sim_data['apparent_solar_zenith'][:, i], index=self._time_index_),
                apparent_azimuth=pd.Series(self.sim_data['solar_azimuth'][:, i], index=self._time_index_),
                axis_tilt=placement.tilt,  # self.placements['tilt'].values,
                axis_azimuth=placement.azimuth,  # self.placements['azimuth'].values,
                max_angle=max_angle,
                backtrack=backtrack,
                gcr=gcr)

            system_tilt[:, i] = tmp['surface_tilt'].values
            system_azimuth[:, i] = tmp['surface_azimuth'].values

            # fix nan values. Why are they there???
            s = np.isnan(system_tilt[:, i])
            system_tilt[s, i] = placement.tilt

            s = np.isnan(system_azimuth[:, i])
            system_azimuth[s, i] = placement.tilt

        self.sim_data['system_tilt'] = system_tilt
        self.sim_data['system_azimuth'] = system_azimuth

        return self

    def determine_angle_of_incidence(self):
        """tracking can be: 'fixed' or 'singleaxis'"""
        assert "apparent_solar_zenith" in self.sim_data
        assert "solar_azimuth" in self.sim_data

        azimuth = self.sim_data.get("system_azimuth", self.placements['azimuth'].values)
        tilt = self.sim_data.get("system_tilt", self.placements['tilt'].values)

        self.sim_data['angle_of_incidence'] = pvlib.irradiance.aoi(
            tilt,
            azimuth,
            self.sim_data['apparent_solar_zenith'],
            self.sim_data['solar_azimuth'])

        return self

    def estimate_plane_of_array_irradiances(self, transposition_model="perez"):

        assert 'apparent_solar_zenith' in self.sim_data
        assert 'solar_azimuth' in self.sim_data
        assert 'direct_normal_irradiance' in self.sim_data
        assert 'global_horizontal_irradiance' in self.sim_data
        assert 'diffuse_horizontal_irradiance' in self.sim_data
        assert 'extra_terrestrial_irradiance' in self.sim_data
        assert 'air_mass' in self.sim_data

        azimuth = self.sim_data.get("system_azimuth", self.placements['azimuth'].values)
        tilt = self.sim_data.get("system_tilt", self.placements['tilt'].values)

        poa = pvlib.irradiance.total_irrad(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            apparent_zenith=self.sim_data['apparent_solar_zenith'],
            azimuth=self.sim_data['solar_azimuth'],
            dni=self.sim_data['direct_normal_irradiance'],
            ghi=self.sim_data['global_horizontal_irradiance'],
            dhi=self.sim_data['diffuse_horizontal_irradiance'],
            dni_extra=self.sim_data['extra_terrestrial_irradiance'],
            airmass=self.sim_data['air_mass'],
            model=transposition_model,)

        for key in poa.keys():
            # This should set: 'poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse', and 'poa_ground_diffuse'

            tmp = poa[key]
            tmp[np.isnan(tmp)] = 0

            self.sim_data[key] = tmp

        assert (self.sim_data['poa_global'] < 1600).all(), "POA is too large"

        return self

    def cell_temperature_from_sandia_method(self):
        assert 'surface_wind_speed' in self.sim_data
        assert 'surface_air_temperature' in self.sim_data
        assert 'poa_global' in self.sim_data

        self.sim_data['cell_temperature'] = solarpower.my_sapm_celltemp(
            self.sim_data['poa_global'],
            self.sim_data['surface_wind_speed'],
            self.sim_data['surface_air_temperature'], )

        return self

    def apply_angle_of_incidence_losses_to_poa(self):
        assert 'poa_direct' in self.sim_data
        assert 'poa_ground_diffuse' in self.sim_data
        assert 'poa_sky_diffuse' in self.sim_data

        tilt = self.sim_data.get("system_tilt", self.placements['tilt'].values)

        self.sim_data["poa_direct"] *= pvlib.pvsystem.physicaliam(self.sim_data['angle_of_incidence'])

        # Effective angle of incidence values from "Solar-Engineering-of-Thermal-Processes-4th-Edition"
        self.sim_data["poa_ground_diffuse"] *= pvlib.pvsystem.physicaliam(90 - 0.5788 * tilt + 0.002693 * np.power(tilt, 2))
        self.sim_data["poa_sky_diffuse"] *= pvlib.pvsystem.physicaliam(59.7 - 0.1388 * tilt + 0.001497 * np.power(tilt, 2))

        self.sim_data['poa_global'] = self.sim_data["poa_direct"] + self.sim_data["poa_ground_diffuse"] + self.sim_data["poa_sky_diffuse"]

        assert (self.sim_data['poa_global'] < 1600).all(), "POA is too large"

        return self

    def configure_cec_module(self, module="WINAICO WSx-240P6"):
        if isinstance(module, str):
            self.register_workflow_parameter("module_name", module)

            if module == "WINAICO WSx-240P6":
                module = pd.Series(dict(
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
                ))
                module.name = "WINAICO WSx-240P6"
            elif module == "LG Electronics LG370Q1C-A5":
                module = pd.Series(dict(
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
                ))
                module.name = "LG Electronics LG370Q1C-A5"
            else:
                # Extract module parameters
                db = pvlib.pvsystem.retrieve_sam("CECMod")
                module = getattr(db, module)

            # Check if we need to add the Desoto parameters
            # defaults for EgRef and dEgdT taken from the note in the docstring for
            #  'pvlib.pvsystem.calcparams_desoto'
            if not "EgRef" in module:
                module['EgRef'] = 1.121
            if not "dEgdT" in module:
                module['dEgdT'] = -0.0002677

        self.module = module

        return self

    def simulate_with_interpolated_single_diode_approximation(self, module="WINAICO WSx-240P6"):
        """
        TODO: Make it work with multiple module definitions
        """
        assert 'poa_global' in self.sim_data
        assert 'cell_temperature' in self.sim_data

        self.configure_cec_module(module)

        sel = self.sim_data['poa_global'] > 0

        poa = self.sim_data['poa_global'][sel]
        cell_temp = self.sim_data['cell_temperature'][sel]

        # Use RectBivariateSpline to speed up simulation, but at the cost of accuracy (should still be >99.996%)
        maxpoa = np.nanmax(poa)

        _poa = np.concatenate([np.logspace(-1, np.log10(maxpoa / 10), 20, endpoint=False),
                               np.linspace(maxpoa / 10, maxpoa, 80)])
        _temp = np.linspace(cell_temp.min(), cell_temp.max(), 100)
        poaM, tempM = np.meshgrid(_poa, _temp)

        sotoParams = pvlib.pvsystem.calcparams_desoto(poa_global=poaM.flatten(),
                                                      temp_cell=tempM.flatten(),
                                                      alpha_isc=self.module.alpha_sc,
                                                      module_parameters=self.module,
                                                      EgRef=self.module.EgRef,
                                                      dEgdT=self.module.dEgdT)

        photoCur, satCur, resSeries, resShunt, nNsVth = sotoParams
        gen = solarpower.mysinglediode(
            photocurrent=photoCur,
            saturation_current=satCur,
            resistance_series=resSeries,
            resistance_shunt=resShunt,
            nNsVth=nNsVth)

        interpolator = RectBivariateSpline(_temp, _poa, gen['p_mp'].reshape(poaM.shape), kx=3, ky=3)
        self.sim_data['module_dc_power_at_mpp'] = np.zeros_like(self.sim_data['poa_global'])
        self.sim_data['module_dc_power_at_mpp'][sel] = interpolator(cell_temp, poa, grid=False)

        interpolator = RectBivariateSpline(_temp, _poa, gen['v_mp'].reshape(poaM.shape), kx=3, ky=3)
        self.sim_data['module_dc_voltage_at_mpp'] = np.zeros_like(self.sim_data['poa_global'])
        self.sim_data['module_dc_voltage_at_mpp'][sel] = interpolator(cell_temp, poa, grid=False)

        self.sim_data['capacity_factor'] = self.sim_data['module_dc_power_at_mpp'] / (self.module.I_mp_ref * self.module.V_mp_ref)

        # Estimate total system generation
        if "capacity" in self.placements.columns:
            self.sim_data['total_system_generation'] = self.sim_data['capacity_factor'] * np.broadcast_to(self.placements.capacity, self._sim_shape_)

        if "modules_per_string" in self.placements.columns and "strings_per_inverter" in self.placements.columns:
            total_modules = self.placements.modules_per_string * \
                self.placements.strings_per_inverter * \
                getattr(self.placements, "number_of_inverters", 1)

            self.sim_data['total_system_generation'] = self.sim_data['module_dc_power_at_mpp'] * np.broadcast_to(total_modules, self._sim_shape_)

        return self

    def apply_inverter_losses(self, inverter, method="sandia", ):
        """method can be: 'sandia' or 'driesse'

        TODO: Make it work with multiplt inverter definitions
        """

        assert 'module_dc_power_at_mpp' in self.sim_data
        assert 'module_dc_voltage_at_mpp' in self.sim_data
        assert hasattr(self, 'module')
        assert "modules_per_string" in self.placements.columns
        assert "strings_per_inverter" in self.placements.columns
        assert not "capacity" in self.placements.columns, "Cannot simultaneously provide 'capacity' and inverter-string parameters"

        if method == "sandia":
            if isinstance(inverter, str):
                db = pvlib.pvsystem.retrieve_sam("SandiaInverter")
                inverter = getattr(db, inverter)

            self.sim_data['inverter_ac_power_at_mpp'] = pvlib.pvsystem.snlinverter(
                v_dc=self.sim_data['module_dc_voltage_at_mpp'] * np.broadcast_to(self.placements.modules_per_string, self._sim_shape_),
                p_dc=self.sim_data['module_dc_power_at_mpp'] * np.broadcast_to(self.placements.modules_per_string * self.placements.strings_per_inverter, self._sim_shape_),
                inverter=inverter)

        elif method == "driesse":
            if isinstance(inverter, str):
                db = pvlib.pvsystem.retrieve_sam("CECInverter")
                inverter = getattr(db, inverter)

            self.sim_data['inverter_ac_power_at_mpp'] = pvlib.pvsystem.adrinverter(
                v_dc=self.sim_data['module_dc_voltage_at_mpp'] * np.broadcast_to(self.placements.modules_per_string, self._sim_shape_),
                p_dc=self.sim_data['module_dc_power_at_mpp'] * np.broadcast_to(self.placements.modules_per_string * self.placements.strings_per_inverter, self._sim_shape_),
                inverter=inverter)

        number_of_inverters = getattr(self.placements, "number_of_inverters", 1)
        self.sim_data['total_system_generation'] = self.sim_data['inverter_ac_power_at_mpp'] * np.broadcast_to(number_of_inverters, self._sim_shape_)

        total_capacity = self.module.I_mp_ref * \
            self.module.V_mp_ref * \
            self.placements.modules_per_string * \
            self.placements.strings_per_inverter * \
            number_of_inverters

        self.sim_data['capacity_factor'] = self.sim_data['total_system_generation'] / np.broadcast_to(total_capacity, self._sim_shape_)

        return self

    # def to_xarray(self, output_netcdf_path=None):
    #     xds = super().to_xarray(_intermediate_dict=True)
