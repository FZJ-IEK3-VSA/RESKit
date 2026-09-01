import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from reskit.dac.data import DATAFOLDER

from ...workflow_manager import WorkflowManager

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
        DACWorkflowManager

        """
        # Do basic workflow construction
        assert all([a in placements.columns for a in ["lon", "lat", "capacity"]]), (
            "Placements must contain the columns lon,lat and capacity"
        )
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

    def load_lt_dac_model_data(self, model: str):
        """
        Function to load the DAC model data of a given model. The model data maps temperature and relative humidity to energy demand, relative productivity and water desorption.
        Description:
        The DAC model data needs columns "T" and "RH" where the temperature (°C) and relative humidity are included. Its a csv file.
        Additionally, the data needs columns:
            -"totalElectricity" where the needed electricity input in MWh/tCO2 is stated at the specified ambient conditions
            -"totalThermal" where the needed heat input in MWh/tCO2 is stated at the specified ambient conditions
            -"relProd" where the relative productivity (i.e. the cpaacity factor) is stated at the specified ambient conditions
            -"waterDesorption" where the produced water is stated in tH2O/tCO2. It might also be possible that this is negative (if water is consumed rather than produced)
        The currently available models:
            -LT_jajjawi: Data from the developed low temperature (solid sorbent) DAC model by Jajjawi et al. [1]. Here, the heat is needed at 90 °C.
            -LT_sendi: Data from the developed low temperature (solid sorbent) DAC model by Sendi et al. [2]. Here, the heat is needed at 110 °C (for steam generation). The original Sendi data has been adapted as described in Wenzel 2025 [3].

        Parameters
        ----------
        model: str
            type of DAC model to use. Valid inputs are: "LT_sendi", "LT_jajjawi" or a path to a csv with DAC model data in the same format as in ./data/

        References
        ----------
        [1] http://dx.doi.org/10.2139/ssrn.5230783
        [2] 10.1016/j.oneear.2022.09.003
        [3] 10.1016/j.adapen.2025.100229

        """
        model_path_dict = {"LT_sendi": "LT_sendi.csv", "LT_jajjawi": "LT_jajjawi.csv"}

        assert model in model_path_dict.keys() or (
            isinstance(model, str) and model.endswith(".csv") and os.path.isfile(model)
        ), (
            f"Invalid model: {model}. Not one of the base models (LT_jajjawi or LT_sendi) and no valid path to an existing csv with custom data."
        )

        if model in model_path_dict.keys():
            path = os.path.join(DATAFOLDER, model_path_dict[model])
        else:
            path = model
        self.dac_data = pd.read_csv(path, index_col=0)

        required_cols = [
            "totalElectricity",
            "totalThermal",
            "relProd",
            "waterDesorption",
        ]
        assert all(col in self.dac_data.columns for col in required_cols), (
            f"Missing columns: {set(required_cols) - set(self.dac_data.columns)}"
        )

    def simulate_lt_dac_model(self, fillMethod: str = "nearest"):
        """
        Simulate the LT DAC (Direct Air Capture) model for the specified plant locations.

        This function interpolates DAC model data to the simulation grid and calculates
        electricity, heat, and water requirements as well as CO2 output for the plants.
        It also handles points outside the convex hull of the DAC data using a specified
        fill method.

        Parameters
        ----------
        fillMethod : str, optional
            Method to fill values for weather conditions outside the convex hull of
            the DAC model data. Options are:
            - "nearest" : use the nearest available datapoint (default)
            - "offTmin" : cut off for temperatures below the DAC data range, use nearest
            for relative humidity

        Raises
        ------
        NotImplementedError
            If a filling method other than "nearest" or "offTmin" is requested.
        """
        if fillMethod not in ["offTmin", "nearest"]:
            raise NotImplementedError(f"Filling method '{fillMethod}' is not implemented. Use 'nearest' or 'offTmin'.")

        # create unique grid as well as interpolators and evaluate for each property:
        properties = ["totalElectricity", "totalThermal", "waterDesorption", "relProd"]
        interpolated_outputs = {}

        for prop in properties:
            # Pivot table for this property
            pivot = self.dac_data.pivot_table(values=prop, index="T", columns="RH")

            # Create interpolator
            interpolator = RegularGridInterpolator(
                points=(pivot.index.values, pivot.columns.values.astype("float64")),
                values=pivot.values,
                bounds_error=False,
            )

            # Evaluate interpolator at simulation data and store into dict
            interpolated_outputs[prop] = interpolator(
                (self.sim_data["surface_air_temperature"], self.sim_data["relative_humidity"])
            )

        if (fillMethod == "offTmin") or (fillMethod == "nearest"):
            # fill points outside the convex hull with "nearest" :
            fill_outputs = {}
            for prop in properties:
                # Create interpolator using nearest-neighbor
                pivot = self.dac_data.pivot_table(values=prop, index="T", columns="RH")
                interpolator = RegularGridInterpolator(
                    points=(pivot.index.values, pivot.columns.values.astype("float64")),
                    values=pivot.values,
                    method="nearest",
                    bounds_error=False,
                    fill_value=None,
                )
                # Evaluate at simulation points
                fill_outputs[prop] = interpolator(
                    (self.sim_data["surface_air_temperature"], self.sim_data["relative_humidity"])
                )
        if fillMethod == "offTmin":
            # fill RH values outside range by nearest and force no operation below/above T bounds by setting relProd=0
            Tmin = self.dac_data["T"].min()
            # fill_outputs holds numpy arrays, not a DataFrame
            fill_outputs["relProd"] = np.where(
                self.sim_data["surface_air_temperature"] < Tmin, 0, fill_outputs["relProd"]
            )

        # combine:
        for prop in properties:
            interpolated_outputs[prop] = np.where(
                np.isnan(interpolated_outputs[prop]), fill_outputs[prop], interpolated_outputs[prop]
            )

        self.sim_data["capacity_factor"] = interpolated_outputs[
            "relProd"
        ]  # the relative productivity for DAC plants equals to the capacity factor for other renewable energy plants, i.e. wind turbines
        self.sim_data["conversion_factor_electricity"] = interpolated_outputs["totalElectricity"]  # MWh_el/t_CO2
        self.sim_data["conversion_factor_heat"] = interpolated_outputs["totalThermal"]  # MWh_th/t_CO2
        self.sim_data["conversion_factor_water"] = interpolated_outputs["waterDesorption"]  # t_H2O/t_CO2

        # Now, besides the conversion factors which are relative to the produced CO2 mass, also simulate the specified plant with the specified capacity:
        self.sim_data["CO2_output"] = self.sim_data["capacity_factor"] * np.array(
            self.placements["capacity"]
        )  # t_CO2/h
        self.sim_data["H2O_output"] = self.sim_data["CO2_output"] * self.sim_data["conversion_factor_water"]  # t_H2O/h
        self.sim_data["electricity_input"] = (
            self.sim_data["CO2_output"] * -self.sim_data["conversion_factor_electricity"]
        )  # MWh_el/h
        self.sim_data["heat_input"] = self.sim_data["CO2_output"] * -self.sim_data["conversion_factor_heat"]  # MWh_th/h

    def simulate_ht_dac_model(self, model: str = "HT_okosun"):
        """
        Simulate the high-temperature (HT), liquid-solvent DAC (Direct Air Capture) model for a given model type.

        This function maps ambient temperature and relative humidity to energy demand,
        relative productivity, and water desorption for the specified DAC model.
        Currently, only the electrified HT DAC model "HT_okosun" is available.

        Parameters
        ----------
        model : str, optional
            Type of DAC model to use. Currently, only "HT_okosun" is implemented.
            Default is "HT_okosun".

        Raises
        ------
        NotImplementedError
            If a DAC model type other than "HT_okosun" is requested.

        Notes
        -----
        The "HT_okosun" model is based on a natural gas-fired HT-DAC system [1],
        adapted to an electrified version as described in [2,3]. The electrified
        DAC model only consumes electricity.

        References
        ----------
        [1] 10.1016/j.apenergy.2022.119895
        [2] 10.3389/fclim.2020.618644
        [3] 10.1016/j.adapen.2025.100229
        """
        # Calculate capture rate, relative productivity and energy (w/o compression)
        if model == "HT_okosun":
            capture_rate = (
                48.8371759783294
                + 0.141875496 * self.sim_data["relative_humidity"]
                + 0.961897256 * self.sim_data["surface_air_temperature"]
                - 0.000550616476 * self.sim_data["relative_humidity"] ** 2
                + 0.00266221049 * self.sim_data["surface_air_temperature"] * self.sim_data["relative_humidity"]
                - 0.00588467947 * self.sim_data["surface_air_temperature"] ** 2
            )  # equation fitted by k.okosun as described in [3]. Describes the share [%] of co2 captured from the incoming air dependent on the ambient conditions. See also [1,2].

            ElecDemand = 7.2082 * capture_rate ** (
                -0.317
            )  # equation fitted by k.okosun as described in [3]. Relates the capture rate to the energy demand.
            relative_productivity = (
                capture_rate / 40 * 527702.4 / 1000000
            )  # equation fitted by k.okosun as described in [3]. Relates the capture rate to the relative productivity.
        else:
            raise NotImplementedError(f"HT-DAC Model of type {model} not implemented.")

        self.sim_data["capacity_factor"] = (
            relative_productivity  # the relative productivity for DAC plants equals to the capacity factor for other renewable energy plants, i.e. wind turbines
        )
        self.sim_data["conversion_factor_electricity"] = -ElecDemand  # MWh_el/t_CO2
        self.sim_data["conversion_factor_heat"] = np.nan  # MWh_th/t_CO2
        self.sim_data["conversion_factor_water"] = np.nan  # tH2O/tCO2

        # Now, besides the conversion factors which are relative to the produced CO2 mass, also simulate the specified plant with the specified capacity:
        self.sim_data["CO2_output"] = self.sim_data["capacity_factor"] * np.array(
            self.placements["capacity"]
        )  # t_CO2/h
        self.sim_data["H2O_output"] = np.nan  # t_H2O/h
        self.sim_data["electricity_input"] = (
            self.sim_data["CO2_output"] * -self.sim_data["conversion_factor_electricity"]
        )  # MWh_el/h
        self.sim_data["heat_input"] = np.nan  # MWh_th/h
