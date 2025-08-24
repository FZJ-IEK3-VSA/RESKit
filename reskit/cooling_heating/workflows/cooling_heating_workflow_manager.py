import pandas as pd
import numpy as np

from collections import OrderedDict
from scipy.interpolate import interp1d

from ...workflow_manager import WorkflowManager


"""

Importing required packages.

"""


class CoolingHeatingWorkflowManager(WorkflowManager):
    def __init__(self, placements):
        """

        __init_(self, placements)

        Initialization of an instance of the generic CoolingHeatingWorkflowManager class.

        Parameters
        ----------
        placements : pandas Dataframe
                     The locations that the simulation should be run for.
                     Columns must include "lon", "lat" (CRS: 4326) and "capacity"
                     -The capacity is the nominal capacity of the Heating/Cooling System

        Returns
        -------
        CoolingHeatingWorkflowManager

        Sources:
        [1] https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html
        [2] https://www.engineeringtoolbox.com/air-specific-heat-capacity-d_705.html

        """

        # Do basic workflow construction
        assert all(
            [a in placements.columns for a in ["lon", "lat", "capacity"]]
        ), "Placements must contain the columns lon,lat and capacity"
        super().__init__(placements)

        # Set thermodynamic data [1] for air density, [2] for air heat capacity
        airData = np.array(
            [
                [
                    0.0002793,
                    0.0002793,
                    0.0002792,
                    0.0002792,
                    0.0002793,
                    0.0002794,
                    0.0002795,
                    0.0002796,
                    0.0002798,
                    0.0002799,
                    0.0002802,
                    0.0002804,
                ],  # heat capacity cp in kWh/(kgK)
                [
                    1.451,
                    1.394,
                    1.341,
                    1.292,
                    1.246,
                    1.204,
                    1.164,
                    1.127,
                    1.093,
                    1.06,
                    1.029,
                    1,
                ],
            ]
        ).T  # density in kg/m3
        self.airData = pd.DataFrame(
            index=[-30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80],
            data=airData,
            columns=["cp", "density"],
        )  # index referes to ambient air temperature

    def calculate_fan_power_air_cooling(
        self,
        temperatureCoolant,
        heatTransferDelta=5,
        efficiencyFan=0.7,
        pressureDropAir=261,
        designTemperature=None,
    ):
        """
        Function to calculate the fan power demand of an air cooling model.

        Parameter:
        temperatureCoolant (float): temperature of the cooling load (lower temperature, if sensible heat transfer) in °C
        heatTransferDelta (float): temperature delta required for heat transfer from air to coolant [K]. defaults to 5.
        efficiencyFan (float): efficiency of the total fan system [0,1]. defaults to 0.7 based on [1]
        pressureDropAir (float): pressure drop of the air through the channels of the frame [Pa]. defaults to 261 Pa based on [2]
        designTemperature (float): if given, the following is only evaluated for the design temperature as air temperature. [°C]

        Sources:
        [1] 10.1016/j.ijhydene.2024.11.381
        [2] http://hdl.handle.net/1853/55674
        """
        if designTemperature:
            airTemp = designTemperature
        else:
            airTemp = self.sim_data["surface_air_temperature"]

        # Build interpolators
        f_cp = interp1d(self.airData.index, self.airData["cp"], kind="linear")
        f_rho = interp1d(self.airData.index, self.airData["density"], kind="linear")
        # interpolate element-wise
        cpAir = f_cp(airTemp)
        densityAir = f_rho(airTemp)

        # Calculate Power demand for 1 kWh of cooling:
        WFan = (
            (
                1
                / (
                    cpAir
                    * (temperatureCoolant - heatTransferDelta - airTemp)
                    * densityAir
                )
            )
            / efficiencyFan
            * pressureDropAir
        )
        WFan = WFan / 1000 / 3600  # Convert J to kWh
        if designTemperature:
            return -WFan
        else:
            WFan[(temperatureCoolant - heatTransferDelta - airTemp) <= 0] = (
                np.inf
            )  # Assign high value if cooling is not possible
            self.sim_data["conversion_factor_fan_electricity"] = -WFan

    def calculate_pump_power_air_cooling(
        self,
        temperatureCoolant,
        heatTransferDelta=5,
        efficiencyPump=0.7,
        pressureDropWater=200000,
        designTemperature=None,
    ):
        """
        Function to calculate the pump power demand of an air cooling model.

        Parameter:
        temperatureCoolant (float): temperature of the cooling load (lower temperature, if sensible heat transfer) in °C
        heatTransferDelta (float): temperature delta required for heat transfer from air to coolant [K]. defaults to 5.
        efficiencyPump (float): efficiency of the total pump system [0,1]. defaults to 0.7 based on [1]
        pressureDropWater (float): pressure drop of the water which is circulated from the site of the heat load to the A-frame [Pa]. defaults to 200000 Pa based on [2]
        designTemperature (float): if given, the following is only evaluated for the design temperature as air temperature. [°C]

        Sources:
        [1] 10.1016/j.ijhydene.2024.11.381
        [2] 10.1016/j.enconman.2020.113610

        Assumptions: Constant density of water (1000 kg/m3) and constant heat capacity of 4.186 kJ/(kgK) = 0.00116 kWh/(kgK)
        """
        if designTemperature:
            airTemp = designTemperature
        else:
            airTemp = self.sim_data["surface_air_temperature"]

        cp = 0.00116  # kWh/(kgK)
        density = 1000  # kg/m3

        # Calculate Power demand for 1 kWh of cooling:
        WPump = (
            (1 / (cp * (temperatureCoolant - heatTransferDelta - airTemp) * density))
            / efficiencyPump
            * pressureDropWater
        )
        WPump = WPump / 1000 / 3600  # Convert J to kWh
        if designTemperature:
            return -WPump
        else:
            WPump[(temperatureCoolant - heatTransferDelta - airTemp) <= 0] = (
                np.inf
            )  # Assign high value if cooling is not possible
            self.sim_data["conversion_factor_pump_electricity"] = -WPump

    def calculate_capacity_factor_air_cooling(
        self,
        designTemperature,
        temperatureCoolant,
        heatTransferDelta=5,
        efficiencyFan=0.7,
        efficiencyPump=0.7,
        pressureDropAir=261,
        pressureDropWater=200000,
    ):
        """
        Function to calculate the capacity factor of an air cooling model.
        Air cooling systems mainly consists of water pumps, fans and th A-frame to transfer the heat from water to the air [1].
        The A-frame is assumed to always transfer heat with the heatTrasferDelta (counter-flow heat exchanger). The cost can be calculated based on equations given in [1].
        Pump and Fan cost dependent on the installed nominal power. However, at varying ambient air temperature, the cost to transfer the same amount of heat would rise, because the needed fan/pump power increases and therefore the needed fan/pump installed capacity increases.
        To account for that, the cost at the design point (design temperature) are calculated and subsequently the cost at all varying air temperatures to transfer the same amount of heat is calculated.
        The capacity factor is then defined as the ratio of cost_at_design_temp/cost_at_air_temp.


        Parameter:
        designTemperature (float): Design ambient temperature of the cooling system
        temperatureCoolant (float): temperature of the cooling load (lower temperature, if sensible heat transfer) in °C
        heatTransferDelta (float): temperature delta required for heat transfer from air to coolant [K]. defaults to 5.
        efficiencyFan (float): efficiency of the total fan system [0,1]. defaults to 0.7
        pressureDropAir (float): pressure drop of the air through the channels of the frame [Pa]. defaults to 261 Pa
        efficiencyPump (float): efficiency of the total pump system [0,1]. defaults to 0.7
        pressureDropWater (float): pressure drop of the water which is circulated from the site of the heat load to the A-frame [Pa]. defaults to 200000 Pa

        Sources:
        [1] 10.1016/j.energy.2015.05.081
        [2] 10.1016/j.enconman.2020.113610
        """

        # At design point:
        PFanDesign = -self.calculate_fan_power_air_cooling(
            temperatureCoolant,
            heatTransferDelta=heatTransferDelta,
            efficiencyFan=efficiencyFan,
            pressureDropAir=pressureDropAir,
            designTemperature=designTemperature,
        )
        CAPEXFanDesign = 2.8 * 12300 * (PFanDesign / 50) ** 0.76  # [1]
        PPumpDesign = -self.calculate_pump_power_air_cooling(
            temperatureCoolant,
            heatTransferDelta=heatTransferDelta,
            efficiencyPump=efficiencyPump,
            pressureDropWater=pressureDropWater,
            designTemperature=designTemperature,
        )
        CAPEXPumpDesign = 2.8 * 3540 * (PPumpDesign) ** 0.71  # [1]

        # At real ambient conditions:
        PFan = -self.sim_data["conversion_factor_fan_electricity"]  # PFan in kW
        CAPEXFan = 2.8 * 12300 * (PFan / 50) ** 0.76  # [1]
        PPump = -self.sim_data["conversion_factor_pump_electricity"]
        CAPEXPump = 2.8 * 3540 * (PPump) ** 0.71  # [1]

        # Air Cooler Cost stays the same in any case:
        alpha = 1.135  # kW/(m2K) [2]
        A = 1 / (alpha * heatTransferDelta)  # needed A-frame size for 1 kW of cooling
        CAPEXAC = 2.8 * 156000 * (A / 200) ** 0.89  # [1]

        CAPEXDesign = CAPEXFanDesign + CAPEXPumpDesign + CAPEXAC
        CAPEX = CAPEXFan + CAPEXPump + CAPEXAC

        self.sim_data["capacity_factor"] = CAPEXDesign / CAPEX

        # set the units of an air cooling system:
        units = {
            "capacity": "kW_th",
            "capacity_factor": "-",
            "conversion_factor_electricity": "kWh_el/kWh_th",
            "conversion_factor_fan_electricity": "kWh_el/kWh_th",
            "conversion_factor_pump_electricity": "kWh_el/kWh_th",
            "electricity_input": "kWh_el",
            "electricity_input_fan": "kWh_el",
            "electricity_input:pump": "kWh_el",
            "cooling_output": "kWh_th",
        }
        self.units = OrderedDict(units)

    def simulate_air_source_heat_pump(
        self, targetTemperature=100, secondLawEfficiency=0.5
    ):
        """
        Function to calculate the coefficient of performance and conversion factors of an air source heat pump.


        Parameter:
        targetTemperature (float): Target temperature at which the heat should be supplied [°C]
        secondLawEfficiency (float): second law efficiency
        """
        self.sim_data["COP"] = (
            (targetTemperature + 273.15)
            / (targetTemperature - self.sim_data["surface_air_temperature"])
            * secondLawEfficiency
        )
        self.sim_data["conversion_factor_electricity"] = (
            -1 / self.sim_data["COP"]
        )  # kWhel/kWhth

        # set the units of an air source heat pump:
        units = {
            "capacity": "kW_th",
            "conversion_factor_electricity": "kWh_el/kWh_th",
            "electricity_input": "kWh_el",
        }
        self.units = OrderedDict(units)
