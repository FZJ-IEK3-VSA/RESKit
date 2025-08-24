import pandas as pd
import numpy as np
import os

from collections import OrderedDict
from scipy.interpolate import griddata


from ...workflow_manager import WorkflowManager
from reskit.dac.data import DATAFOLDER

"""

Importing required packages.

"""


class DACWorkflowManager(WorkflowManager):
    def __init__(self, placements):
        """

        __init_(self, placements)

        Initialization of an instance of the generic DACWorkflowManager class.

        Parameters
        ----------
        placements : pandas Dataframe
                     The locations that the simulation should be run for.
                     Columns must include "lon", "lat" (CRS: 4326) and "capacity"
                     -The capacity is the nominal capacity of the DAC plant in tCO2/h

        Returns
        -------
        DACWorkflorManager

        """

        # Do basic workflow construction
        assert all(
            [a in placements.columns for a in ["lon", "lat", "capacity"]]
        ), "Placements must contain the columns lon,lat and capacity"
        super().__init__(placements)

        units = {
            "capacity": "t_CO2/h",
            "capacity_factor": "-",
            "conversion_factor_electricity": "MWh_el/t_CO2",
            "conversion_factor_heat": "MWh_heat/t_CO2",
            "conversion_factor_water": "t_H2O/t_CO2",
            "CO2_output": "t_CO2",
            "H2O_output": "t_H2O",
            "electricity_input": "MWh_el",
            "heat_input": "MWh_heat",
        }
        self.units = OrderedDict(units)

    def calculate_relative_humidity(self):
        """
        Function to calculate the relative humidity from dewpoint temperature and air temperature using the Sonntag formula.

        References:
        -------------
        [1] https://www.npl.co.uk/resources/q-a/dew-point-and-relative-humidity
        """

        def calculate_vapor_pressure(temperature):
            """
            Function to calculate the vapor pressure from the temperature

            temperature: temperature in °C
            """
            temperature_Kelvin = temperature + 273.15
            vapor_pressure = np.exp(
                -6096.9385 / temperature_Kelvin
                + 21.2409642
                - 2.711193 * 10**-2 * temperature_Kelvin
                + 1.673952 * 10**-5 * temperature_Kelvin**2
                + 2.433502 * np.log(temperature_Kelvin)
            )  # vapor pressure [1]
            return vapor_pressure

        self.sim_data["relative_humidity"] = (
            calculate_vapor_pressure(self.sim_data["surface_dew_temperature"])
            / calculate_vapor_pressure(self.sim_data["surface_air_temperature"])
            * 100
        )  # [1]
        del self.sim_data["surface_dew_temperature"]

    def load_lt_dac_model_data(self, model):
        """
        Function to load the DAC model data of a given model. The model data maps temperature and relative humidity to energy demand, relative productivity and water desorption.

        Parameter:
        model (str): type of DAC model to use. Valid inputs are: "LT_sendi", "LT_jajjawi".

        Description:
        The DAC model data needs columns "T" and "RH" where the temperature (°C) and relative humidity are included.
        Additionally, the data needs columns:
            -"totalElectricity" where the needed electricity input in MWh/tCO2 is stated at the specified ambient conditions
            -"totalThermal" where the needed heat input in MWh/tCO2 is stated at the specified ambient conditions
            -"relProd" where the relative productivity (i.e. the cpaacity factor) is stated at the specified ambient conditions
            -"waterDesorption" where the produced water is stated in tH2O/tCO2. It might also be possible that this is negative (if water is consumed rather than produced)
        The currently available models:
            -LT_jajjawi: Data from the developed low temperature (solid sorbent) DAC model by Jajjawi et al. [1]. Here, the heat is needed at 90 °C.
            -LT_sendi: Data from the developed low temperature (solid sorbent) DAC model by Sendi et al. [2]. Here, the heat is needed at 110 °C (for steam generation). The original Sendi data has been adapted as described in Wenzel 2025 [3].

        References:
        --------------
        [1] http://dx.doi.org/10.2139/ssrn.5230783
        [2] 10.1016/j.oneear.2022.09.003
        [3] 10.1016/j.adapen.2025.100229

        """

        model_path_dict = {"LT_sendi": "LT_sendi.csv", "LT_jajjawi": "LT_jajjawi.csv"}
        path = os.path.join(DATAFOLDER, model_path_dict[model])
        self.dac_data = pd.read_csv(path, index_col=0)

    def simulate_lt_dac_model(self, fillMethod="nearest"):
        """
        Function to simulate the LT DAC model.

        Parameter:
        fillMethod (str): method to use when the weather conditions are not inside the hull of the DAC model weather data.
        -nearest: use the nearest available datapoint
        -offTmin: cut off for temperature ranges, nearest for relative humidity
        default: "nearest"

        """

        elec = griddata(
            (self.dac_data["T"], self.dac_data["RH"]),
            self.dac_data["totalElectricity"],
            (
                self.sim_data["surface_air_temperature"],
                self.sim_data["relative_humidity"],
            ),
            method="linear",
        )
        thermal = griddata(
            (self.dac_data["T"], self.dac_data["RH"]),
            self.dac_data["totalThermal"],
            (
                self.sim_data["surface_air_temperature"],
                self.sim_data["relative_humidity"],
            ),
            method="linear",
        )
        water = griddata(
            (self.dac_data["T"], self.dac_data["RH"]),
            self.dac_data["waterDesorption"],
            (
                self.sim_data["surface_air_temperature"],
                self.sim_data["relative_humidity"],
            ),
            method="linear",
        )
        relProd = griddata(
            (self.dac_data["T"], self.dac_data["RH"]),
            self.dac_data["relProd"],
            (
                self.sim_data["surface_air_temperature"],
                self.sim_data["relative_humidity"],
            ),
            method="linear",
        )

        if fillMethod == "nearest":
            # fill points outside the convex hull with "nearest" :
            elecFill = griddata(
                (self.dac_data["T"], self.dac_data["RH"]),
                self.dac_data["totalElectricity"],
                (
                    self.sim_data["surface_air_temperature"],
                    self.sim_data["relative_humidity"],
                ),
                method="nearest",
            )
            thermalFill = griddata(
                (self.dac_data["T"], self.dac_data["RH"]),
                self.dac_data["totalThermal"],
                (
                    self.sim_data["surface_air_temperature"],
                    self.sim_data["relative_humidity"],
                ),
                method="nearest",
            )
            waterFill = griddata(
                (self.dac_data["T"], self.dac_data["RH"]),
                self.dac_data["waterDesorption"],
                (
                    self.sim_data["surface_air_temperature"],
                    self.sim_data["relative_humidity"],
                ),
                method="nearest",
            )
            relProdFill = griddata(
                (self.dac_data["T"], self.dac_data["RH"]),
                self.dac_data["relProd"],
                (
                    self.sim_data["surface_air_temperature"],
                    self.sim_data["relative_humidity"],
                ),
                method="nearest",
            )
        elif fillMethod == "offTmin":
            # fill RH values outside range by nearest and force no operation below/above T bounds by setting relProd=0
            elecFill = griddata(
                (self.dac_data["T"], self.dac_data["RH"]),
                self.dac_data["totalElectricity"],
                (
                    self.sim_data["surface_air_temperature"],
                    self.sim_data["relative_humidity"],
                ),
                method="nearest",
            )
            thermalFill = griddata(
                (self.dac_data["T"], self.dac_data["RH"]),
                self.dac_data["totalThermal"],
                (
                    self.sim_data["surface_air_temperature"],
                    self.sim_data["relative_humidity"],
                ),
                method="nearest",
            )
            waterFill = griddata(
                (self.dac_data["T"], self.dac_data["RH"]),
                self.dac_data["waterDesorption"],
                (
                    self.sim_data["surface_air_temperature"],
                    self.sim_data["relative_humidity"],
                ),
                method="nearest",
            )
            relProdFill = griddata(
                (self.dac_data["T"], self.dac_data["RH"]),
                self.dac_data["relProd"],
                (
                    self.sim_data["surface_air_temperature"],
                    self.sim_data["relative_humidity"],
                ),
                method="nearest",
            )
            Tmin = self.dac_data["T"].min()
            relProdFill[self.sim_data["surface_air_temperature"] < Tmin] = 0
        else:
            raise NotImplementedError(f"Filling Method: {fillMethod} not implemented.")

        # combine:
        elec = np.where(np.isnan(elec), elecFill, elec)
        thermal = np.where(np.isnan(thermal), thermalFill, thermal)
        water = np.where(np.isnan(water), waterFill, water)
        relProd = np.where(np.isnan(relProd), relProdFill, relProd)

        self.sim_data["capacity_factor"] = (
            relProd  # the relative productivity for DAC plants equals to the capacity factor for other renewable energy plants, i.e. wind turbines
        )
        self.sim_data["conversion_factor_electricity"] = elec
        self.sim_data["conversion_factor_heat"] = thermal
        self.sim_data["conversion_factor_water"] = water

        # Now, besides the conversion factors which are relative to the produced CO2 mass, also simulate the specified plant with the specified capacity:
        self.sim_data["CO2_output"] = self.sim_data["capacity_factor"] * np.array(
            self.placements["capacity"]
        )
        self.sim_data["H2O_output"] = (
            self.sim_data["CO2_output"] * self.sim_data["conversion_factor_water"]
        )
        self.sim_data["electricity_input"] = (
            self.sim_data["CO2_output"]
            * -self.sim_data["conversion_factor_electricity"]
        )
        self.sim_data["heat_input"] = (
            self.sim_data["CO2_output"] * -self.sim_data["conversion_factor_heat"]
        )

    def simulate_ht_dac_model(self, model):
        """
        Function to simulate the HT (high temeperature, liquid solvent)-DAC model data of a given model. The model data maps temperature and relative humidity to energy demand, relative productivity and water desorption.

        Parameter:
        model (str): type of DAC model to use. Valid inputs are: "HT_okosun"

        Description:
        The currently available models:
            -HT_okosun: This model is derived based on a natural gas fired HT-DAC model described in [1]. The data has been adapted to an electrified system as described in [2]. The description is detailed in [3]. The electrified DAC model only has an electricity input.


        References:
        --------------
        [1] 10.1016/j.apenergy.2022.119895
        [2] 10.3389/fclim.2020.618644
        [3] 10.1016/j.adapen.2025.100229
        """

        # Calculate capture rate, relative productivity and energy (w/o compression)
        capture_rate = (
            48.8371759783294
            + 0.141875496 * self.sim_data["relative_humidity"]
            + 0.961897256 * self.sim_data["surface_air_temperature"]
            - 0.000550616476 * self.sim_data["relative_humidity"] ** 2
            + 0.00266221049
            * self.sim_data["surface_air_temperature"]
            * self.sim_data["relative_humidity"]
            - 0.00588467947 * self.sim_data["surface_air_temperature"] ** 2
        )

        ElecDemand = 7.2082 * capture_rate ** (-0.317)
        relative_productivity = capture_rate / 40 * 527702.4 / 1000000

        self.sim_data["capacity_factor"] = (
            relative_productivity  # the relative productivity for DAC plants equals to the capacity factor for other renewable energy plants, i.e. wind turbines
        )
        self.sim_data["conversion_factor_electricity"] = -ElecDemand
        self.sim_data["conversion_factor_heat"] = np.nan
        self.sim_data["conversion_factor_water"] = np.nan

        # Now, besides the conversion factors which are relative to the produced CO2 mass, also simulate the specified plant with the specified capacity:
        self.sim_data["CO2_output"] = self.sim_data["capacity_factor"] * np.array(
            self.placements["capacity"]
        )
        self.sim_data["H2O_output"] = np.nan
        self.sim_data["electricity_input"] = (
            self.sim_data["CO2_output"]
            * -self.sim_data["conversion_factor_electricity"]
        )
        self.sim_data["heat_input"] = np.nan
