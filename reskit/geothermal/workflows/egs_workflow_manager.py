from distutils.log import warn
import numpy as np
import pandas as pd
import xarray as xr
import os
import geokit as gk
import time
from datetime import datetime

from ..data.gringarten import gringarten


class EGS_workflowmanager:

    SECONDS_PER_YEAR = 365 * 24 * 3600
    USD2EUR = 0.88  # EUR
    rho_water = 1000  # kg/m^3
    cp_water = 4182  # J/kg/K

    def __init__(self, placements) -> None:
        """Workflow manager for EGS calcualtions

        Parameters
        ----------
        placements : pd.DataFrame
            placements, contains at least "lat", "lon" columns in EPSG:4326
        """

        self.placements = placements
        assert "lat" in placements.columns
        assert "lon" in placements.columns
        self.sim_data = {}

    def loadPlantData(self, configuration: str, manual_values={}):

        configuration = configuration.lower()
        if not configuration in ["doublette", "triplette"]:
            raise ValueError(
                "Currently only 'doublette' and 'triplette' are supported types."
            )

        def eta_plant(temp_degC):
            """calcualte efficiency based on the protocol from beardsmore

            Parameters
            ----------
            temp : number
                temperature in Â°C

            Returns
            -------
            efficiency of an average EGS plant
                in [1]
            """
            return 0.00052 * temp_degC + 0.032

        data_triplette = {
            "CF": 0.9,  # 1
            "N_wells": 3,  # 1
            "maxDepth_m": 7000,  # m
            "WACC": 0.08,  # %
            "lifetime_a": 30,  # a
            "minRockTemperature_degC": 150,  # Â°C
            "opex_fix_perc_invest_per_a": 0.02,  # %Capex/a
            "reservoir_size_m3": 1.5e9,  # m^3
            "recovery_factor": 0.14,  # 1
            "dT_drawdown": 10,  # K
            "T_inj": 80,  # degC
            "eta_plant": eta_plant,  # 1, f(T[Â°C])
            "eta_pump_1": 0.675,  # 1
            "productivity_(l_per_s)/bar": 2,  # l/s/bar
            "Vdot_total_m3_per_s": 100 * 1e-3,  # m^3/s = 1E-3 l/s
            "x_m": 1000,  # m
            "y_m": 1500,  # m
            "z_m": 1000,  # m
            "x_ED_1": 8,
        }

        data_doublette = {
            "CF": 0.9,  # 1
            "N_wells": 2,  # 1
            "maxDepth_m": 7000,  # m
            "WACC": 0.08,  # %
            "lifetime_a": 30,  # a
            "minRockTemperature_degC": 150,  # Â°C
            "opex_fix_perc_invest_per_a": 0.02,  # %Capex/a
            "reservoir_size_m3": 1.0e9,  # m^3
            "recovery_factor": 0.14,  # 1
            "dT_drawdown": 10,  # K
            "T_inj": 80,  # degC
            "eta_plant": eta_plant,  # 1, f(T[Â°C])
            "eta_pump_1": 0.675,  # 1
            "productivity_(l_per_s)/bar": 2,  # l/s/bar
            "Vdot_total_m3_per_s": 100 * 1e-3,  # m^3/s = 1E-3 l/s
            "x_m": 1000,  # m
            "y_m": 1000,  # m
            "z_m": 1000,  # m
            "x_ED_1": 8,
        }

        if configuration == "doublette":
            self.data = data_doublette
        elif configuration == "triplette":
            self.data = data_triplette
        else:
            raise ValueError(
                "configuration not properly defined. Something went wrong."
            )

        for key in manual_values:
            if key in self.data:
                self.data[key] = manual_values[key]
            else:
                raise ValueError("Manual set values not found.")

        self.data["n_production_wells_1"] = self.data["N_wells"] - 1

    def loadData(self, vars: list, source: str, newVarNamesDict: dict = None):
        """loads the variables from source. Must be a NC file. Cannot be depth depending-

        Parameters
        ----------
        vars : list
            List of varnames to load
        source : str
            path to *.nc file
        newVarNames : dict, optional
            list of new Variable names, by default None
        """
        assert os.path.isfile(source)
        assert isinstance(vars, list)
        assert newVarNamesDict is None or isinstance(newVarNamesDict, dict)

        ds = xr.open_dataset(source)

        # dims
        def set_dims(**vals):
            # loop all kwargs
            for name in vals:
                value = vals[name]
                if hasattr(self, name):
                    # if value is already set
                    assert np.allclose(
                        self.__getattribute__(name), value
                    ), f"dimensions do not align, pls check dim: {name} in file: {source}"
                else:
                    # set value if not existant
                    setattr(self, name, value)
                    # self.__setattr__(name=value)

        set_dims(latsGlobal=ds.lat.values, lonsGlobal=ds.lon.values)
        points = self.placements[["lon", "lat"]].values

        for var in vars:
            if "depth" in ds.dims:
                ds_flat = ds.drop_dims("depth")
            else:
                ds_flat = ds
            data_var_depth = self._EGS_NC4_to_raster(nc4_obj=ds_flat, varname=var)
            values = self._extractPointsFromRaster(points, data_var_depth)
            assert ~np.isnan(values).any()

            self.placements[var] = values

        # renaim certain vars
        if newVarNamesDict is not None:
            self.placements = self.placements.rename(newVarNamesDict, axis=1)

    def loadDataAllDepths(self, vars, source):
        """loads all variables as an array depending on its depth

        Parameters
        ----------
        vars : list
            list of variables, eg: ['Global_EGS_HeatSust',' Global_EGS_HeatTech', 'Global_EGS_PowerSust', 'Global_EGS_PowerTech']
        sourceEGSpotentials : str
            path to the location of the input data

        Returns
        -------
        updates self.sim_data
            adds the variables to the self.sim_data container as np.ndarray(dtype=np.float66)
        """
        assert os.path.isfile(source)

        ds = xr.open_dataset(source)

        # get depths
        def set_dims(**vals):
            # loop all kwargs
            for name in vals:
                value = vals[name]
                if hasattr(self, name):
                    # if value is already set
                    assert np.allclose(
                        self.__getattribute__(name), value
                    ), f"dimensions do not align, pls check dim: {name} in file: {source}"
                else:
                    # set value if not existant
                    setattr(self, name, value)
                    # self.__setattr__(name=value)

        depths = ds.depth.values
        set_dims(depths=depths, latsGlobal=ds.lat.values, lonsGlobal=ds.lon.values)

        # depths = ds.depth.values
        # self.depths = depths
        # self.latsGlobal =  ds.lat.values
        # self.lonsGlobal =  ds.lon.values

        # convert coordinates to np.array
        points = self.placements[["lon", "lat"]].values

        # get data
        shape = (
            len(self.depths),
            len(self.placements),
        )

        for var in vars:
            data_var = np.ndarray(shape=shape, dtype=np.float64)
            data_var[:, :] = np.nan

            for depth in depths:

                # create gdal dataset
                ds_depth = ds.loc[{"depth": depth}]
                data_var_depth = self._EGS_NC4_to_raster(nc4_obj=ds_depth, varname=var)
                values = self._extractPointsFromRaster(points, data_var_depth)
                assert ~np.isnan(values).any()

                # store var inside container
                depth_index = list(depths).index(depth)
                data_var[depth_index, :] = values

            # save variable to sim_data
            self.sim_data[var] = data_var

    def _extractPointsFromRaster(self, points, data_var_depth):
        # interpolate Values
        values = gk.raster.interpolateValues(
            source=data_var_depth,
            points=points,
            pointSRS=gk.srs.loadSRS(4326),
            mode="linear-spline",
        )

        # handle nans with nearest #TODO: interpolation when not all values from window range 1 are given
        nan_index = np.isnan(values)
        if sum(nan_index) > 0:
            values_near = gk.raster.interpolateValues(
                source=data_var_depth,
                points=points,
                pointSRS=gk.srs.loadSRS(4326),
                mode="near",
            )
            values[nan_index] = values_near[nan_index]

        # handle nans with nearest
        nan_index = np.isnan(values)
        if sum(nan_index) > 0:

            def find_nearest_value(data, x_offset, y_offset):
                x_mid = int((data.shape[0] + 1) / 2)
                y_mid = int((data.shape[1] + 1) / 2)

                mid_value = data[x_mid, y_mid]
                if not np.isnan(mid_value):
                    return mid_value

                mean_value_window1 = np.nanmean(
                    data[x_mid - 1 : x_mid + 2, x_mid - 1 : x_mid + 2]
                )
                if not np.isnan(mean_value_window1):
                    return mean_value_window1

                mean_value_window2 = np.nanmean(
                    data[x_mid - 2 : x_mid + 3, x_mid - 2 : x_mid + 3]
                )
                if not np.isnan(mean_value_window2):
                    return mean_value_window2

                mean_value_window3 = np.nanmean(
                    data[x_mid - 3 : x_mid + 4, x_mid - 3 : x_mid + 4]
                )
                if not np.isnan(mean_value_window3):
                    return mean_value_window3

                return np.nan  # dummy value

            values_near = gk.raster.interpolateValues(
                source=data_var_depth,
                points=points,
                pointSRS=gk.srs.loadSRS(4326),
                mode="func",
                winRange=3,
                func=find_nearest_value,
            )
            values[nan_index] = values_near[nan_index]
        return values

    def AssignReservoirVolumeSize(self):
        self.placements["ReservoirSize_m^3"] = self.data["reservoir_size_m3"]

    def __calulateRockVolumeShare(self):
        """calcualtes the share of the placement reservoir to the discretized volume from the input data

        Parameters
        ----------
        reservoirSize : float, optional
            Volume of the plant specific reservoir in m^3, by default 1E9 m^3
        """
        EARTHRADIUS = 6371e3  # m

        dLat = self.latsGlobal[1] - self.latsGlobal[0]
        dY = 2 * np.pi * EARTHRADIUS * dLat / 360  # m

        dLon = self.lonsGlobal[1] - self.lonsGlobal[0]
        r_lat = EARTHRADIUS * np.cos(np.deg2rad(self.latsGlobal))
        dX = 2 * np.pi * r_lat * dLon / 360  # m

        dDepth = self.depths[1] - self.depths[0]  # m

        volume_lat = dY * dX * dDepth  # m^3
        volumes_dict = dict(zip(self.latsGlobal, volume_lat))

        lats_placements_rounded = list((self.placements.lat.values - 0.5).round() + 0.5)
        DiscretizedRockVolume_placments = np.array(
            [volumes_dict[lat] for lat in lats_placements_rounded]
        )
        share = self.data["reservoir_size_m3"] / DiscretizedRockVolume_placments

        self.placements["share_1"] = share

    def __calculatePlacementHeat(self):

        variables = list(self.sim_data.keys())
        if "temperature" in variables:
            variables.remove("temperature")

        # variables = []
        for var in variables:
            varname_new = f"{var}_placement"
            self.sim_data[varname_new] = (
                self.sim_data[var] * self.placements["share_1"].values
            )

    def ___calculatePlantOutput(self):
        # calculation?
        self.sim_data["P_Plant_nom_UNITHERE"] = (
            self.sim_data["Global_EGS_PowerTech"] * 1 / self.data["CF"]
        )

    def VolumeMethod(self):
        """calculate the enthalpy from the temperature"""
        self.sim_data_VM = {}
        # define rock properties
        rho_rock = 2550  # kg/m^3
        cp_rock = 1000  # J/kg/K
        dT_drawdown = self.data["dT_drawdown"]  # K
        T_inj = self.data["T_inj"]  # Â°C
        eta_plant = self.data["eta_plant"]

        # total enthalpy
        Enth = (
            rho_rock
            * cp_rock
            * self.data["reservoir_size_m3"]
            * (
                self.sim_data["temperature"]
                - self.placements["surface_temperature"].values
            )
        )
        self.sim_data_VM["Total_thermal_energy_PJ"] = Enth / 1e15

        # useable enthalpy
        R_TD = dT_drawdown / (
            self.sim_data["temperature"] - self.placements["surface_temperature"].values
        )
        Enth_useable = Enth * R_TD * self.data["recovery_factor"]  # J
        Enth_useable[self.sim_data["temperature"] < T_inj] = 0

        # average heatflow over lifetime
        Qdot_out = Enth_useable / self.data["lifetime_a"] / (self.SECONDS_PER_YEAR)  # W
        self.sim_data_VM["Qdot_out_VM_MW"] = Qdot_out / 1e6

        # average power
        T_out = (
            self.sim_data["temperature"] - dT_drawdown / 2
        )  # water outlet temperature
        eta = eta_plant(T_out)  # average temperature over lifetime
        P_out = Qdot_out * eta
        self.sim_data_VM["P_out_VM_MW"] = P_out / 1e6

        # resource useage
        Tdot_K_per_a = dT_drawdown / self.data["lifetime_a"]  # K/a
        dT_useable = (
            self.sim_data["temperature"] - self.data["minRockTemperature_degC"]
        )  # K
        resource_use_time = dT_useable / Tdot_K_per_a  # a

        # debugging
        cp_water = 4182  # J/kgK
        mdot_water = Qdot_out / (cp_water * (T_out - T_inj))  # kg/s

        self.sim_data_VM["mdot_water_VM_kg_per_s"] = mdot_water
        self.sim_data_VM["mdot_water_VM_kg_per_s_per_well"] = (
            mdot_water / self.data["n_production_wells_1"]
        )

        self.sim_data_VM["dT_active_res_VM_K"] = dT_drawdown
        self.sim_data_VM["dT_total_res_VM_K"] = (
            dT_drawdown * self.data["recovery_factor"]
        )
        self.sim_data_VM["recovery_fac_amb_VM_1"] = Enth_useable / Enth

        self.sim_data_VM["resourceUseTime_VM_a"] = resource_use_time

        self.sim_data_VM["temperature_VM_degC"] = self.sim_data["temperature"]
        self.sim_data_VM["T_Rock_abandon_VM_degC"] = (
            self.sim_data["temperature"] - self.sim_data_VM["dT_total_res_VM_K"]
        )
        self.sim_data_VM["T_Water_out_VM_degC"] = self.sim_data_VM[
            "temperature_VM_degC"
        ]
        pass

    def GringartenMethodFixedT(self):
        """[summary]"""
        # assumptions
        x_ED = 2  # set, so that the gringarten recovery factor also leads to 14%
        T_D = self.data["dT_drawdown"] / (
            self.sim_data["temperature"] - self.data["T_inj"]
        )
        x = y = z = 1000  # m reservoir size
        t = self.data["lifetime_a"] * self.SECONDS_PER_YEAR  # s
        t_D = (
            0.5  # 10.5 #dimensionless time, see Augustine2016 Fig 2 for TD=0.05, xed=4
        )

        # rock properties
        rho_rock = 2550  # kg/m^3
        cp_rock = 1000  # J/kg/K
        k_R = 2.5  # W/mK #TODO: real location specific values here
        warn(
            "K_R is set to a default value. this needs to be adapted for the final Calulations"
        )

        q = (
            np.sqrt(
                (t_D * k_R * rho_rock * cp_rock)
                / ((self.rho_water * self.cp_water) ** 2 * t)
            )
            * y
            * z
        )  # m^3/s?, inj. water volume flow per fracture, see Augustine2016 eq(2)
        x_E = (
            x_ED * k_R * y * z / (self.rho_water * self.cp_water * q)
        )  # m?, fracture spacing, see Augustine2016 eq(3)
        n = x / (
            2 * x_E
        )  # 1, number of fractures, can be decimal as theoretical calcualtion nevertheless
        Q = q * n  # l/s, total inj. water volume flow

        #######################################s
        # validated until here

        T_W_out = self.sim_data["temperature"] - self.data["dT_drawdown"]
        dT_W = T_W_out - self.data["T_inj"]

        Q_dot = Q * self.rho_water * self.cp_water * dT_W
        eta = self.data["eta_plant"](T_W_out)
        P_out = Q_dot * eta

        pass

    def GringartenMethodFixeVdot(self):
        """calculates the Gringarten solution for a given fracture configuration"""
        assert self.data["lifetime_a"] % 1 == 0  # check if its a natural number

        Vdot_total = self.data["Vdot_total_m3_per_s"]  # m^3/s = 1E-3 l/s
        x = self.data["x_m"]  # m
        y = self.data["y_m"]  # m
        z = self.data["z_m"]  # m
        x_ED = self.data["x_ED_1"]  # 1

        grin = gringarten(
            Vdot_total=Vdot_total,
            x=x,
            y=y,
            z=z,
            x_ED=x_ED,
        )

        num_of_timesteps = 1000  # self.data['lifetime_a']
        grin.getDimlessTime(
            np.linspace(1, num_of_timesteps, num_of_timesteps) * self.SECONDS_PER_YEAR
        )
        grin.getGringartenCurve()
        grin.getWaterTemp(self.sim_data["temperature"], self.data["T_inj"])
        ans = grin.getEGSProps(timestep=self.data["lifetime_a"])
        resource_use_time_a = grin.getResourceUseTime(
            T_abandon=self.data["minRockTemperature_degC"]
        )

        # calc eta
        eta = self.data["eta_plant"](
            ans["T_Water_out"]
        )  # average temperature over lifetime
        P_out = ans["Qdot_water"] * eta

        # water flow:
        mdot_water_GR_kg_per_s = ans["mdot_water"] * self.data["n_production_wells_1"]

        # save vars
        self.sim_data_GR = {}
        self.sim_data_GR["Qdot_out_GR_MW"] = ans["Qdot_water"] / 1e6
        self.sim_data_GR["P_out_GR_MW"] = P_out / 1e6
        self.sim_data_GR["mdot_water_GR_kg_per_s"] = mdot_water_GR_kg_per_s
        self.sim_data_GR["mdot_water_GR_kg_per_s_per_well"] = ans["mdot_water"]

        self.sim_data_GR["dT_active_res_GR_K"] = -1
        self.sim_data_GR["dT_total_res_GR_K"] = ans["dT_Rock_avrg"]
        self.sim_data_GR["recovery_fac_amb_GR_1"] = ans["R_total"]

        self.sim_data_GR["T_Rock_abandon_GR_degC"] = ans["T_Rock_avrg"]
        self.sim_data_GR["T_Water_out_GR_degC"] = ans["T_Water_out"]

        self.sim_data_GR["resourceUseTime_GR_a"] = resource_use_time_a

        self.sim_data_GR["temperature_GR_degC"] = self.sim_data["temperature"]

    def SustainableHeat(self):
        assert "qdot_sust_W_per_m2" in self.placements.columns

        qdot_sust_W_per_m2 = self.placements["qdot_sust_W_per_m2"].values

        # get absolute sustainable heatflow
        delta_depth = self.depths[1] - self.depths[0]  # m
        Area_horizonztal = self.data["reservoir_size_m3"] / delta_depth  # m
        Qdot_sust_W = Area_horizonztal * qdot_sust_W_per_m2  # a
        # adjust shape to fit placements * depth
        Qdot_sust_W = np.einsum("i,j", np.ones(len(self.depths)), Qdot_sust_W)

        # get plant conversion efficiency
        T_water_out = self.sim_data["temperature"]  # degC

        eta_plant = self.data["eta_plant"](T_water_out)

        # get the power
        P_el_MW = Qdot_sust_W * 1e-6 * eta_plant

        # get water mass flow
        mdot_water_kg_per_s = Qdot_sust_W / (
            self.rho_water * self.cp_water * (T_water_out - self.data["T_inj"])
        )
        mdot_water_kg_per_s_per_well = (
            mdot_water_kg_per_s / self.data["n_production_wells_1"]
        )

        # save vars
        self.sim_data_SU = {}

        self.sim_data_SU["Qdot_out_SU_MW"] = Qdot_sust_W / 1e6
        self.sim_data_SU["P_out_SU_MW"] = P_el_MW
        self.sim_data_SU["mdot_water_SU_kg_per_s"] = mdot_water_kg_per_s
        self.sim_data_SU["mdot_water_SU_kg_per_s_per_well"] = (
            mdot_water_kg_per_s_per_well
        )

        self.sim_data_SU["dT_active_res_SU_K"] = 0
        self.sim_data_SU["dT_total_res_SU_K"] = 0
        self.sim_data_SU["recovery_fac_amb_SU_1"] = -1

        self.sim_data_SU["T_Rock_abandon_SU_degC"] = T_water_out
        self.sim_data_SU["T_Water_out_SU_degC"] = T_water_out

        self.sim_data_SU["resourceUseTime_SU_a"] = np.inf

        self.sim_data_SU["temperature_SU_degC"] = self.sim_data["temperature"]

    def calculatePumpLosses(self, method="default", techMethod=None):
        """[summary]

        Parameters
        ----------
        method : str, optional
            [description], by default 'default'
        techMethod : [type], optional
            [description], by default None
        """
        # get the data from self
        sim_data_techmethod = getattr(self, techMethod)
        tech_method_short = self._getTechMethodShort(techMethod)

        Vdot_m3_per_s_per_well = (
            1
            / self.rho_water
            * sim_data_techmethod[f"mdot_water_{tech_method_short}_kg_per_s_per_well"]
        )
        productivity_m3_per_s_per_Pa = (
            self.data["productivity_(l_per_s)/bar"] * 1e-3 / 1e5
        )  # Pa/(m^3/s)
        detaP = Vdot_m3_per_s_per_well / productivity_m3_per_s_per_Pa  # Pa

        P_pump = detaP * Vdot_m3_per_s_per_well / self.data["eta_pump_1"]
        P_pump_MW = P_pump / 1e6

        sim_data_techmethod[f"P_Pump_{tech_method_short}_MW"] = P_pump_MW
        sim_data_techmethod[f"P_out_net_{tech_method_short}_MW"] = (
            sim_data_techmethod[f"P_out_{tech_method_short}_MW"] - P_pump_MW
        )

    def calculateCosts(self, method="default", techMethod=None):
        """calcualte the CAPEX cost for the plant"""

        # get the data from self
        sim_data_techmethod = getattr(self, techMethod)
        tech_method_short = self._getTechMethodShort(techMethod)

        if method == "aghahosseini2020":
            # Capex
            # Well
            CAPEX_Well_MUSD = (
                1.72 * 10**-7 * self.depths**2 + 2.3 * 10**-3 * self.depths - 0.62
            )
            CAPEX_Well_MUSD = np.tile(CAPEX_Well_MUSD, (len(self.placements), 1)).T
            # Plant
            P_nom = (
                sim_data_techmethod[f"P_out_{tech_method_short}_MW"] / self.data["CF"]
            )
            CAPEX_Plant_MUSD = (
                (750 + 1125 * np.exp(-0.006115 * (P_nom - 5))) * P_nom / 1e3
            )
            # Other
            CAPEX_Stim_MUSD = 2.5
            CAPEX_Distr_MUSD = 50 * P_nom / 1e3
            CAPEX_Expl_MUSD = 1.12 * (1 + 0.6 * CAPEX_Well_MUSD)
            # add up
            CAPEX_Total_MUSD = (
                CAPEX_Well_MUSD * self.data["N_wells"]
                + CAPEX_Plant_MUSD
                + CAPEX_Stim_MUSD
                + CAPEX_Distr_MUSD
                + CAPEX_Expl_MUSD
            )

            pass
        elif method == "default":
            # Capex
            # Well
            # Lukawski https://aip.scitation.org/doi/10.1063/1.4865575
            CAPEX_Well_MUSD_Lukawski = (
                1.72 * 10**-7 * self.depths**2 + 2.3 * 10**-3 * self.depths - 0.62
            )
            # SAM Intermediate 1	Vertical Open Hole	Larger Diameter	$ 3,243,076 	0.189267288	293.4517365	1326526.313, CAPEX_Well_USD = c1*x^2 + c2 *x + c3, x in m
            # https://www.energy.gov/eere/geothermal/geovision
            CAPEX_Well_MUSD_SAM_Intermed1 = (
                1326526.313 + 293.4517365 * self.depths + 0.189267288 * self.depths**2
            ) / 1e6

            CAPEX_Well_MUSD = np.tile(
                CAPEX_Well_MUSD_SAM_Intermed1, (len(self.placements), 1)
            ).T
            # Plant
            P_nom = (
                sim_data_techmethod[f"P_out_{tech_method_short}_MW"] / self.data["CF"]
            )
            CAPEX_Plant_MUSD = (
                1560 * P_nom / 1e3
            )  # 1560 EUR/kW from 2006_Heidinger-et-al
            # Other
            CAPEX_Stim_MUSD = 2.5
            CAPEX_Pump_MUSD = (
                1720
                * sim_data_techmethod[f"P_Pump_{tech_method_short}_MW"]
                * self.data["n_production_wells_1"]
                / 1e3
            )
            CAPEX_Expl_MUSD = 1.85
            # add up
            CAPEX_Total_MUSD = (
                CAPEX_Well_MUSD * self.data["N_wells"]
                + CAPEX_Plant_MUSD
                + CAPEX_Stim_MUSD
                + CAPEX_Pump_MUSD
                + CAPEX_Expl_MUSD
            )
        else:
            CAPEX_Total_MUSD = np.nan

        # calc annuity
        self.data["annuity"] = (
            (1 + self.data["WACC"]) ** self.data["lifetime_a"] * self.data["WACC"]
        ) / ((1 + self.data["WACC"]) ** self.data["lifetime_a"] - 1)

        # get yearly costs
        CAPEX_Total_MUSD_per_a = CAPEX_Total_MUSD * self.data["annuity"]
        OPEX_fix_MUSD_per_a = CAPEX_Total_MUSD * self.data["opex_fix_perc_invest_per_a"]
        OPEX_var_MUSD_per_a = 0

        # sum it up to TOTEX
        TOTEX_MUSD_per_a = (
            CAPEX_Total_MUSD_per_a + OPEX_fix_MUSD_per_a + OPEX_var_MUSD_per_a
        )
        sim_data_techmethod[f"TOTEX_MUSD_{tech_method_short}_per_a"] = TOTEX_MUSD_per_a
        pass

    def calculateLCOE(self, techMethod):
        sim_data_techmethod = getattr(self, techMethod)
        tech_method_short = self._getTechMethodShort(techMethod)

        # self.sim_data['LCOE_sust_MUSD_per_unkwn'] = self.sim_data['TOTEX_MUSD_per_a'] / (self.sim_data['Global_EGS_PowerSust'] * 8760)
        sim_data_techmethod[f"LCOE_gross_{tech_method_short}_EUR_per_kWh"] = (
            sim_data_techmethod[f"TOTEX_MUSD_{tech_method_short}_per_a"]
            * self.USD2EUR
            * 1e6
            / (sim_data_techmethod[f"P_out_{tech_method_short}_MW"] * 1e3 * 8760)
        )
        sim_data_techmethod[f"LCOE_{tech_method_short}_EUR_per_kWh"] = (
            sim_data_techmethod[f"TOTEX_MUSD_{tech_method_short}_per_a"]
            * self.USD2EUR
            * 1e6
            / (sim_data_techmethod[f"P_out_net_{tech_method_short}_MW"] * 1e3 * 8760)
        )

    def getOptDepth(self, techMethod):
        """gets the optimal depth value based on the lowest LCOE"""

        sim_data_techmethod = getattr(self, techMethod)
        tech_method_short = self._getTechMethodShort(techMethod)

        LCOE_considerable = sim_data_techmethod[f"LCOE_{tech_method_short}_EUR_per_kWh"]
        # filter by temperature if using the volume method
        LCOE_considerable[
            self.sim_data["temperature"] < self.data["minRockTemperature_degC"]
        ] = np.inf

        # filter by max depth:
        maxDepth = self.data["maxDepth_m"]
        indexDepths = self.depths <= maxDepth
        LCOE_considerable = LCOE_considerable[indexDepths]

        # filter errorneous values:
        LCOE_considerable[np.isnan(LCOE_considerable)] = np.inf
        LCOE_considerable[LCOE_considerable <= 0] = np.inf

        # non Eligible Locations:
        notEligible = (LCOE_considerable == np.inf).all(axis=0)

        # select
        argminDepth = np.argmin(LCOE_considerable, axis=0)
        optDepths = self.depths[argminDepth]
        # correct by notEligible
        argminDepth = argminDepth.astype(np.float)
        optDepths = optDepths.astype(np.float)
        argminDepth[notEligible] = np.nan
        optDepths[notEligible] = np.nan

        # set argminoptDepts
        sim_data_techmethod[f"opt_depth_{tech_method_short}_m"] = optDepths
        sim_data_techmethod[f"argmin_opt_depth_{tech_method_short}_m"] = argminDepth
        sim_data_techmethod[f"notEligible_{tech_method_short}"] = notEligible

    def getValuesAtOptDepth(self, techMethod):
        """[summary]"""
        sim_data_techmethod = getattr(self, techMethod)
        tech_method_short = self._getTechMethodShort(techMethod)

        def getOptimalValue(self, mat, argminOptDepth):
            """returns the values at the optimal depth"""
            placement = np.arange(
                0, len(argminOptDepth)
            )  # np.arange(0, len(self.placements))
            return mat[argminOptDepth, placement]

        # write optimal values into placements(results!) based on their type
        for varname in sim_data_techmethod.keys():
            var = sim_data_techmethod[varname]

            assert varname not in self.placements.columns

            # irrelevant for storing
            if "argmin_opt_depth" in varname:
                continue
            if "notEligible" in varname:
                continue
            # if its opt_dept, just write it
            if "opt_depth" in varname:
                self.placements[varname] = var
            # if its a constant value, just write it
            elif isinstance(var, int) or isinstance(var, float):
                self.placements[varname] = var
            # if its an depth depending matrix, select the right depth depending value
            elif isinstance(var, np.ndarray):
                notEligible = sim_data_techmethod[f"notEligible_{tech_method_short}"]
                argminOptDepth = sim_data_techmethod[
                    f"argmin_opt_depth_{tech_method_short}_m"
                ]

                # get eligible Values
                argminOptDepthEligible = argminOptDepth[~notEligible].astype(np.int)
                varEligible = var[:, ~notEligible]
                valEligible = getOptimalValue(self, varEligible, argminOptDepthEligible)

                # get all Values
                var = np.nan * np.ones(len(self.placements))
                var[~notEligible] = valEligible
                var[notEligible] = np.nan

                self.placements[varname] = var

    def getRegenerationTime(self, techMethod):

        sim_data_techmethod = getattr(self, techMethod)
        tech_method_short = self._getTechMethodShort(techMethod)

        Q_out_Wa = (
            sim_data_techmethod[f"Qdot_out_{tech_method_short}_MW"]
            * 1e6
            * self.data["lifetime_a"]
        )  # Wa = Watt*year
        delta_depth = self.depths[1] - self.depths[0]  # m
        regeneration_time = Q_out_Wa / (
            self.data["reservoir_size_m3"]
            / delta_depth
            * self.placements["qdot_sust_W_per_m2"].values
        )  # a

        sim_data_techmethod[f"regeneration_time_{tech_method_short}_a"] = (
            regeneration_time
        )

    def saveOutput(self, savepath=None, deepsave=False):
        """saved to nc4 or shape file or csv to savepath

        Parameters
        ----------
        savepath : str, optional
            String to a specific file path (folder + filename).
            Can be shape, nc4, or csv. If None, printing
        deepsave : bool, optional
            If True, saves more variables. Defaults to False
        """

        def _convertToXr(placements):
            """convert placements to NC4 xarray obj

            Parameters
            ----------
            placements : pd.Dataframe
                placement file from EGS_workflowmanager

            Returns
            -------
            xr.Dataset
                Dataset with placemnt variables
            """
            ds_out = xr.Dataset(
                coords={
                    "placements": np.arange(len(placements)),
                },
                attrs=dict(
                    description=f"RESkit simulation results for EGS.",
                    timestamp=f" Generated at: {datetime.now()} ",
                ),
            )
            for varname in placements.columns:
                values = placements[varname].values
                ds_out[varname] = (["placements"], values)

            return ds_out

        if savepath is None:
            # No savepath given
            print("No valid file type specified. Returning nc4 obj")
            return _convertToXr(self.placements)

        else:
            # save the file
            assert isinstance(savepath, str)
            folder = os.path.dirname(savepath)
            file = os.path.basename(savepath)
            _, filetype = os.path.splitext(savepath)

            os.makedirs(folder, exist_ok=True)

            if filetype.lower() == ".shp":
                # do shapefile
                self.placements["geom"] = self.placements[["lon", "lat"]].apply(
                    lambda x: gk.geom.point(x[0], x[1]), axis=1
                )
                gk.vector.createVector(self.placements, savepath)
            elif filetype.lower() == ".nc4":
                # do netcdf4
                placements_nc4 = _convertToXr(self.placements)
                placements_nc4.to_netcdf(savepath)
            elif filetype.lower() == ".csv":
                # saving as excel:
                self.placements.to_csv(savepath)
            elif not savepath is None:
                # saving as excel:
                savepath = os.path.join(folder, file, "csv")
                self.placements.to_csv()

            print("Results saved to:", savepath)

            # save all techMethod datasets
            if deepsave:
                savepathDeepsave = savepath.replace(filetype, "_<TECHMETHOD>.xlsx")
                techMethods = self._getTechMethods()
                techMethods.append("sim_data")
                for techMethod in techMethods:
                    savepathDeepsaveFileld = savepathDeepsave.replace(
                        "<TECHMETHOD>", str(techMethod)
                    )
                    # save as excel:
                    techMethodDict = getattr(self, techMethod)
                    with pd.ExcelWriter(savepathDeepsaveFileld) as writer:
                        for key in techMethodDict:
                            var = techMethodDict[key]
                            if isinstance(var, int) or isinstance(var, float):
                                df_temp = pd.DataFrame()
                                df_temp[0] = [var]
                            else:
                                df_temp = pd.DataFrame(var)
                            df_temp.to_excel(writer, sheet_name=key)

    def _getTechMethods(self):
        # get tech_methods
        techMethods = []
        for i in self.__dir__():
            if "sim_data_" in i:
                techMethods.append(i)

        return techMethods

    def _getTechMethodShort(self, techMethod):
        return techMethod[-2:]

    def _EGS_NC4_to_raster(self, nc4_obj, varname):
        """extracts one depth layer from an variable and returns it as an raster file

        Parameters
        ----------
        nc4_obj : xr.dataset
            [description]
        varname : str
            name of the variable

        Returns
        -------
        gdal.Dataset
            Gdal tiff representation of the given variable
        """
        assert list(nc4_obj.dims) == ["lat", "lon"]

        lons = nc4_obj.lon.values
        lats = nc4_obj.lat.values
        pixelWidth = float((lons[-1] - lons[0]) / (len(lons) - 1))
        pixelHeight = float((lats[-1] - lats[0]) / (len(lats) - 1))

        bounds = [
            min(lons) - pixelWidth / 2,  # xMin
            min(lats) - pixelHeight / 2,  # yMin
            max(lons) + pixelWidth / 2,  # xMax
            max(lats) + pixelHeight / 2,  # yMax
        ]

        srs = gk.srs.loadSRS(4326)

        data = nc4_obj[varname].values
        data = np.flip(data, axis=0)

        raster = gk.raster.createRaster(
            bounds,
            output=None,
            pixelWidth=pixelWidth,
            pixelHeight=pixelHeight,
            dtype=np.float64,
            srs=srs,
            compress=True,
            noData=np.nan,
            overwrite=True,
            data=data,
        )

        # import matplotlib.pyplot as plt
        # gk.drawRaster(raster)
        # plt.savefig('raster.png')

        return raster

    @staticmethod
    def volumeMethodSingular(
        reservoir_size_m3,
        dT_drawdown,
        recovery_factor,
        temperature,
        surface_temperature=10,
        lifetime_a=30,
        depth=3000,
        n_wells=2,
        T_inj=80,
    ):
        """calculate the enthalpy from the temperature"""

        USD2EUR = 0.88
        ### Physical
        # define rock properties
        rho_rock = 2550  # kg/m^3
        cp_rock = 1000  # J/kg/K

        def eta_plant(temp_degC):
            """calcualte efficiency based on the protocol from beardsmore

            Parameters
            ----------
            temp : number
                temperature in Â°C

            Returns
            -------
            efficiency of an average EGS plant
                in [1]
            """
            return 0.00052 * temp_degC + 0.032

        # total enthalpy
        Enth = (
            rho_rock * cp_rock * reservoir_size_m3 * (temperature - surface_temperature)
        )

        # useable enthalpy
        R_TD = dT_drawdown / (temperature - surface_temperature)
        Enth_useable = Enth * R_TD * recovery_factor  # J
        if temperature < T_inj:
            Enth_useable = 0

        # average heatflow over lifetime
        Qdot_out = Enth_useable / lifetime_a / (365 * 24 * 3600)  # W

        # average power
        T_out = temperature - dT_drawdown / 2  # water outlet temperature
        eta = eta_plant(T_out)  # average temperature over lifetime
        P_out = Qdot_out * eta

        # capacity
        P_nom = P_out / 0.9

        # debugging

        cp_water = 4182  # J/kgK
        mdot_water = Qdot_out / (cp_water * (T_out - T_inj))  # kg/s

        Qdot_out_MW = Qdot_out / 1e6
        mdot_water_kg_per_s = mdot_water
        P_out_MW = P_out / 1e6

        ### COST

        CAPEX_Well_MUSD = 1.72 * 10**-7 * depth**2 + 2.3 * 10**-3 * depth - 0.62
        CAPEX_Plant_MUSD = (
            (750 + 1125 * np.exp(-0.006115 * (P_nom / 1e6 - 5))) * P_nom / 1e9
        )
        CAPEX_Stim_MUSD = 2.5
        CAPEX_Distr_MUSD = 50 * P_nom / 1e9
        CAPEX_Expl_MUSD = 1.12 * (1 + 0.6 * CAPEX_Well_MUSD)

        CAPEX_Total_MUSD = (
            CAPEX_Well_MUSD * n_wells
            + CAPEX_Plant_MUSD
            + CAPEX_Stim_MUSD
            + CAPEX_Distr_MUSD
            + CAPEX_Expl_MUSD
        )

        # calc annuity
        annuity = ((1 + 0.08) ** lifetime_a * 0.08) / ((1 + 0.08) ** lifetime_a - 1)

        # get yearly costs
        CAPEX_Total_MUSD_per_a = CAPEX_Total_MUSD * annuity
        OPEX_fix_MUSD_per_a = CAPEX_Total_MUSD * 0.02
        OPEX_var_MUSD_per_a = 0

        # sum it up
        TOTEX_MUSD_per_a = (
            CAPEX_Total_MUSD_per_a + OPEX_fix_MUSD_per_a + OPEX_var_MUSD_per_a
        )

        ### LCOE
        LCOE_tech_EUR_per_kWh = (
            TOTEX_MUSD_per_a * USD2EUR * 1e6 / (P_out_MW * 1e3 * 8760)
        )
        pass

        # recocvery factor
        heat_in_place = reservoir_size_m3 * 2.55e6 * (temperature)
        heat_used = Qdot_out_MW * 1e6 * 30 * 365 * 24 * 3600
        recovery_factor_amb = heat_used / heat_in_place

        recovery_factor_amb_2 = recovery_factor * dT_drawdown / temperature

        dT_drawdown_total_reservoir = heat_used / (reservoir_size_m3 * 2.55e6)

        names = {
            "reservoir_size_m3": reservoir_size_m3,
            "dT_drawdown": dT_drawdown,
            "dT_drawdown_total_reservoir": dT_drawdown_total_reservoir,
            "temperature": temperature,
            "Qdot_out_MW": Qdot_out_MW,
            "mdot_water_kg_per_s": mdot_water_kg_per_s,
            "P_out_MW": P_out_MW,
            "LCOE_tech_EUR_per_kWh": LCOE_tech_EUR_per_kWh,
            "recovery_factor_amb": recovery_factor_amb,
        }
        return pd.Series(names)


if __name__ == "__main__":
    print("\nThis is not an executable file. Pls run EGSworkflow(args)\n")
