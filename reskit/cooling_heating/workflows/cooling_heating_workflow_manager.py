import pandas as pd
import numpy as np
import xarray as xr

from collections import OrderedDict
from scipy.interpolate import interp1d

from reskit.workflow_manager import WorkflowManager
from reskit.util.specific_humidity import calculate_specific_humidity


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
        assert all([a in placements.columns for a in ["lon", "lat", "capacity"]]), (
            "Placements must contain the columns lon,lat and capacity"
        )
        super().__init__(placements)

        # Set thermodynamic data [1] for air density, [2] for air heat capacity
        airData = np.array(
            [
                [
                    0.0002799,
                    0.0002797,
                    0.0002795,
                    0.0002794,
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
                    1.739,
                    1.657,
                    1.582,
                    1.514,
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
            index=[
                -70,
                -60,
                -50,
                -40,
                -30,
                -20,
                -10,
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
            ],
            data=airData,
            columns=["cp", "density"],
        )  # index refers to ambient air temperature

        self.evaporationCoolingData = pd.DataFrame(
            data=[[1006, 1860, 2.257 * 10**6, 4184]], columns=["cp_air", "cp_vapor", "evaporationHeat", "cp_water"]
        )  # J/(kg*K), J/(kg*K), J/(kg), J/(kg*K)

    def calculate_approach_evaporative_cooling(
        self,
        temperatureCoolant: float | int,
        heatTransferDelta: float | int,
        efficiencyCoolingTower: float | int,
    ):
        """
        Calculate the approach temperature for an evaporative-cooling system.


        Parameters
        ----------
        temperatureCoolant : float | int
            Temperature of the cooling load (lower temperature if sensible heat transfer) in °C.
        heatTransferDelta : float | int
            Temperature difference required for heat transfer from air to coolant [K]
        efficiencyCoolingTower : float | int
            Efficiency of the cooling tower system [0, 1]

        Returns
        -------
        float or np.ndarray
            Approach temperature for the specified weather conditions.

        References
        ----------
        [1] 10.1016/j.ijhydene.2024.11.381
        """
        # Calculate T_CChot
        T_CChot = temperatureCoolant - heatTransferDelta
        # Calculate T_CCcold using wet bulb temperature approximation
        T_CCcold = T_CChot - efficiencyCoolingTower * (T_CChot - self.sim_data["wet_bulb_temperature"])
        # Calculate approach temperature
        approach_temperature = T_CCcold - self.sim_data["wet_bulb_temperature"]

        self.sim_data["approach_temperature_evaporative_cooling"] = approach_temperature

    def calculate_water_losses_evaporative_cooling(
        self,
        temperatureCoolant: float | int,
        heatTransferDelta: float | int,
        efficiencyCoolingTower: float | int,
        factorDriftLosses: float | int = 0.001,
        typical_cycles_blowdown: int = 5,
    ):
        """
        Calculate the water losses for an evaporative-cooling system.


        Parameters
        ----------
        temperatureCoolant : float | int
            Temperature of the cooling load (lower temperature if sensible heat transfer) in °C.
        heatTransferDelta : float | int
            Temperature difference required for heat transfer from air to coolant [K]
        efficiencyCoolingTower : float | int
            Efficiency of the cooling tower system [0, 1]
        factorDriftLosses : float | int
            Drift losses by small water droplets carried away by the exhaust air. Defaults to 0.001. [1]
        typical_cycles_blowdown: int
            after how many cycles the blowdown occurs to prevent accumulation of impurities. Defaults to 5. [2]

        Returns
        -------
        float or np.ndarray
            Specific water losses of evaporative cooling towers (per kWh of cooling load) for the specified weather conditions.

        References
        ----------
        [1] 10.1016/j.ijhydene.2024.11.381
        [2] 10.1016/j.enconman.2020.113610
        """
        specific_humidity_inlet = calculate_specific_humidity(
            self.sim_data["surface_air_temperature"],
            self.sim_data["relative_humidity"]
            / 100,  # relative humidity between 0,1 needed to calcaulte the specific humidity
        )
        specific_humidity_outlet = calculate_specific_humidity(
            self.sim_data["wet_bulb_temperature"] + self.sim_data["approach_temperature_evaporative_cooling"],
            np.full_like(self.sim_data["wet_bulb_temperature"], 1.0),
        )

        specific_enthalpy_inlet = self.evaporationCoolingData["cp_air"][0] * (
            self.sim_data["surface_air_temperature"] + 273.15
        ) + specific_humidity_inlet * (
            self.evaporationCoolingData["cp_vapor"][0] * (self.sim_data["surface_air_temperature"] + 273.15)
            + self.evaporationCoolingData["evaporationHeat"][0]
        )  # J/kg
        specific_enthalpy_outlet = self.evaporationCoolingData["cp_air"][0] * (
            (self.sim_data["wet_bulb_temperature"] + 273.15) + self.sim_data["approach_temperature_evaporative_cooling"]
        ) + specific_humidity_outlet * (
            self.evaporationCoolingData["cp_vapor"][0]
            * (
                (self.sim_data["wet_bulb_temperature"] + 273.15)
                + self.sim_data["approach_temperature_evaporative_cooling"]
            )
            + self.evaporationCoolingData["evaporationHeat"][0]
        )  # J/kg

        # calculate needed air mass specific for 1 kWh cooling load
        air_mass = 1 / (
            (specific_enthalpy_outlet - specific_enthalpy_inlet) / 3600000
        )  # enthalpy from J/kg to kWh/kg --> air_mass in kg (per kWh)
        evaporation_loss = air_mass * (specific_humidity_outlet - specific_humidity_inlet)
        self.sim_data["specific_mass_evaporation_loss"] = evaporation_loss

        # drift losses:
        water_mass = 1 / (
            self.evaporationCoolingData["cp_water"] * heatTransferDelta / 3600000
        )  # calcualte total water mass, enthalpy from J/kg to kWh/kg --> water_mass in kg (per kWh)
        drift_losses = water_mass * factorDriftLosses
        self.sim_data["specific_mass_drift_loss"] = drift_losses

        # blowdown losses (periodic discharge of water to prevent accumulation of impurities):
        blowdown_losses = evaporation_loss / (typical_cycles_blowdown - 1)
        self.sim_data["specific_mass_blowdown_loss"] = blowdown_losses

        self.sim_data["conversion_factor_water"] = -(
            evaporation_loss + drift_losses[0] + blowdown_losses
        )  # corresponds to the water losses (therefore negative)

        units = {
            "capacity": "kW_th",
            "conversion_factor_water": "kg_H2O/kWh_th",
            "wet_bulb_temperature": "°C",
            "approach_temperature_evaporative_cooling": "K",
            "specific_mass_evaporation_loss": "kg_H2O/kWh_th",
            "specific_mass_drift_loss": "kg_H2O/kWh_th",
            "specific_mass_blowdown_loss": "kg_H2O/kWh_th",
        }
        self.units = OrderedDict(units)

    def calculate_fan_power_air_cooling(
        self,
        temperatureCoolant: float | int,
        heatTransferDelta: float | int = 5,
        efficiencyFan: float | int = 0.7,
        pressureDropAir: float | int = 261,
        designTemperature: float | int = None,
    ):
        """
        Calculate the fan power demand for an air-cooling system.

        This method computes the electrical power required by the fan to transfer heat
        from the air to a coolant, based on the air properties and the cooling load
        temperature. It can evaluate either for a given design air temperature or for
        the time series of ambient air temperatures stored in `self.sim_data`.

        Parameters
        ----------
        temperatureCoolant : float | int
            Temperature of the cooling load (lower temperature if sensible heat transfer) in °C.
        heatTransferDelta : float | int, optional
            Temperature difference required for heat transfer from air to coolant [K]. Default is 5.
        efficiencyFan : float | int, optional
            Efficiency of the fan system [0, 1]. Default is 0.7.
        pressureDropAir : float | int, optional
            Pressure drop of air through the channels of the cooling frame [Pa]. Default is 261.
        designTemperature : float | int, optional
            If specified, the calculation is only evaluated at this air temperature [°C].
            Default is None.

        Returns
        -------
        float or np.ndarray
            Fan power demand in kWh per kWh of cooling. Returns a single value if
            `designTemperature` is provided, or a time series array otherwise.

        Notes
        -----
        - Uses linear interpolation of air specific heat (`cp`) and density (`rho`) from `self.airData`.
        - Assigns `np.inf` to any time step where the temperature difference is insufficient for cooling.
        - Stores the time series result in `self.sim_data["conversion_factor_fan_electricity"]` if `designTemperature` is None.

        References
        ----------
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
            (1 / (cpAir * (temperatureCoolant - heatTransferDelta - airTemp) * densityAir))
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
        temperatureCoolant: float | int,
        heatTransferDelta: float | int = 5,
        efficiencyPump: float | int = 0.7,
        pressureDropWater: float | int = 200000,
        designTemperature: float | int = None,
    ):
        """
        Calculate the pump power demand for an air-cooling system.

        This method computes the electrical power required by the pump to circulate
        coolant for an air-cooling system, based on the cooling load temperature,
        pressure drop, and pump efficiency. It can evaluate either for a given design
        air temperature or for the time series of ambient air temperatures stored in
        `self.sim_data`.

        Parameters
        ----------
        temperatureCoolant : float | int
            Temperature of the cooling load (lower temperature if sensible heat transfer) in °C.
        heatTransferDelta : float | int, optional
            Temperature difference required for heat transfer from air to coolant [K]. Default is 5.
        efficiencyPump : float | int, optional
            Efficiency of the pump system [0, 1]. Default is 0.7.
        pressureDropWater : float | int, optional
            Pressure drop of the water circuit between the heat load and the cooling frame [Pa]. Default is 200000.
        designTemperature : float | int, optional
            If specified, the calculation is only evaluated at this air temperature [°C].
            Default is None.

        Returns
        -------
        float or np.ndarray
        Pump power demand in kWh per kWh of cooling. Returns a single value if
        `designTemperature` is provided, or a time series array otherwise.

        Notes
        -----
        - Assumes constant water density of 1000 kg/m³ and specific heat capacity of 4.186 kJ/(kg·K) = 0.00116 kWh/(kg·K).
        - Assigns `np.inf` to any time step where the temperature difference is insufficient for cooling.
        - Stores the time series result in `self.sim_data["conversion_factor_pump_electricity"]` if `designTemperature` is None.

        References
        ----------
        [1] 10.1016/j.ijhydene.2024.11.381
        [2] 10.1016/j.enconman.2020.113610
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

    def calculate_relative_cost_factor_air_cooling(
        self,
        designTemperature: float | int,
        temperatureCoolant: float | int,
        heatTransferDelta: float | int = 5,
        efficiencyFan: float | int = 0.7,
        efficiencyPump: float | int = 0.7,
        pressureDropAir: float | int = 261,
        pressureDropWater: float | int = 200000,
    ):
        """
        Calculate the relative (air temperature dependent) cost factor of an air-cooling system.

        The air-cooling system consists of fans, water pumps, and an A-frame heat exchanger
        that transfers heat from water to air. The A-frame is assumed to always transfer
        heat with the specified `heatTransferDelta`. Fan and pump costs depend on the
        installed nominal power, which varies with ambient air temperature. The relative cost
        factor is defined as the ratio of cost at the design temperature to the cost at
        actual ambient temperatures. To cool the same amount of heat at an air temperature higher than the design air temperature, the CAPEX would rise, since pump and fan capacity would increase.

        Parameters
        ----------
        designTemperature : float | int
            Design ambient temperature of the cooling system [°C].
        temperatureCoolant : float | int
            Temperature of the cooling load (lower temperature if sensible heat transfer) [°C].
        heatTransferDelta : float | int, optional
            Temperature difference required for heat transfer from air to coolant [K]. Default is 5.
        efficiencyFan : float | int, optional
            Efficiency of the fan system [0, 1]. Default is 0.7.
        efficiencyPump : float | int, optional
            Efficiency of the pump system [0, 1]. Default is 0.7.
        pressureDropAir : float | int, optional
            Pressure drop of air through the channels of the cooling frame [Pa]. Default is 261.
        pressureDropWater : float | int, optional
            Pressure drop of the water circuit from the heat load to the A-frame [Pa]. Default is 200000.

        Returns
        -------
        None

        The method stores the calculated relative cost factor in `self.sim_data["relative_cost_factor"]`
        and updates the `self.units` dictionary with air-cooling system units.

        Notes
        -----
        - Fan and pump CAPEX are calculated based on nominal power at the design temperature and
        scaled according to ambient conditions [1].
        - A-frame cost is assumed constant and independent of ambient temperature [2].
        - The relative cost factor reflects the relative increase in cost at varying ambient conditions
        compared to the design point.

        References
        ----------
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

        self.sim_data["relative_cost_factor"] = CAPEXDesign / CAPEX

        # set the units of an air cooling system:
        units = {
            "capacity": "kW_th",
            "relative_cost_factor": "-",
            "capacity_factor": "-",
            "conversion_factor_electricity": "kWh_el/kWh_th",
            "conversion_factor_fan_electricity": "kWh_el/kWh_th",
            "conversion_factor_pump_electricity": "kWh_el/kWh_th",
            "electricity_input": "kWh_el",
            "electricity_input_fan": "kWh_el",
            "electricity_input_pump": "kWh_el",
            "cooling_output": "kWh_th",
        }
        self.units = OrderedDict(units)

    def calculate_capacity_factor_air_cooling(
        self,
        designTemperature: float | int,
        temperatureCoolant: float | int,
        heatTransferDelta: float | int = 5,
        efficiencyFan: float | int = 0.7,
        efficiencyPump: float | int = 0.7,
        pressureDropAir: float | int = 261,
        pressureDropWater: float | int = 200000,
    ):
        """
        Calculate the capacity factor of an air-cooling system.

        The air-cooling system can only provide the cooling load if the ambient air
        temperature is lower than the design temperature. If the temperature is above the design temperature,
        pumps and fans are designed too small to provide the necessary flows and the system can only provide less cooling.
        The reduction in cooling can be calculated based on the ratio of the design power and the theoretically required
        power for both, fans and pumps individually. The minimum of these ratios is the capacity factor.
        If the temperature rises above the coolant temperature minus the heat transfer delta, the system needs to shut off.

        Parameters
        ----------
        designTemperature : float | int
            Design ambient temperature of the cooling system [°C].
        temperatureCoolant : float | int
            Temperature of the cooling load (lower temperature if sensible heat transfer) [°C].
        heatTransferDelta : float | int, optional
            Temperature difference required for heat transfer from air to coolant [K]. Default is 5.
        efficiencyFan : float | int, optional
            Efficiency of the fan system [0, 1]. Default is 0.7.
        efficiencyPump : float | int, optional
            Efficiency of the pump system [0, 1]. Default is 0.7.
        pressureDropAir : float | int, optional
            Pressure drop of air through the channels of the cooling frame [Pa]. Default is 261.
        pressureDropWater : float | int, optional
            Pressure drop of the water circuit from the heat load to the A-frame [Pa]. Default is 200000.

        Returns
        -------
        None
            The method stores the calculated capacity factor in `self.sim_data["capacity_factor"]'.
        """
        # At design point:
        PFanDesign = self.calculate_fan_power_air_cooling(
            temperatureCoolant,
            heatTransferDelta=heatTransferDelta,
            efficiencyFan=efficiencyFan,
            pressureDropAir=pressureDropAir,
            designTemperature=designTemperature,
        )
        PPumpDesign = self.calculate_pump_power_air_cooling(
            temperatureCoolant,
            heatTransferDelta=heatTransferDelta,
            efficiencyPump=efficiencyPump,
            pressureDropWater=pressureDropWater,
            designTemperature=designTemperature,
        )

        self.sim_data["capacity_factor"] = xr.where(
            self.sim_data["surface_air_temperature"] <= designTemperature,
            1,  # Case 1: below design temperature, the system can always provide enough cooling.
            xr.where(
                self.sim_data["surface_air_temperature"] < (temperatureCoolant - heatTransferDelta),
                # Case 2: between design and shut off temperature. The system can not provide sufficient cooling. Its limited by:
                np.minimum(
                    PPumpDesign / self.sim_data["conversion_factor_pump_electricity"],  # either the pump
                    PFanDesign / self.sim_data["conversion_factor_fan_electricity"],  # or the fan
                ),
                0.0,  # Case 3: too hot. The system needs to shut off.
            ),
        )

    def simulate_air_source_heat_pump(
        self,
        targetTemperature: float | int = 100,
        secondLawEfficiency: float | int = 0.5,
    ):
        """
        Simulate an air-source heat pump and calculate its coefficient of performance (COP) and conversion factors.

        The method computes the COP based on the ambient air temperature and the target supply
        temperature of the heat pump. It also calculates the electricity conversion factor per
        unit of thermal energy delivered.

        Parameters
        ----------
        targetTemperature : float | int, optional
            Target temperature at which the heat should be supplied [°C]. Default is 100.
        secondLawEfficiency : float | int, optional
            Second law efficiency of the heat pump [0,1]. Default is 0.5.

        Returns
        -------
        None
            The calculated COP and conversion factors are stored in `self.sim_data`.
            The `self.units` dictionary is also updated to reflect the units of the heat pump.

        Notes
        -----
        - The electricity conversion factor is calculated as -1/COP [kWh_el/kWh_th].
        - Units set include thermal capacity (`kW_th`), electricity conversion factor (`kWh_el/kWh_th`),
        and electricity input (`kWh_el`).
        """
        self.sim_data["COP"] = (
            (targetTemperature + 273.15)
            / (targetTemperature - self.sim_data["surface_air_temperature"])
            * secondLawEfficiency
        )
        self.sim_data["conversion_factor_electricity"] = -1 / self.sim_data["COP"]  # kWhel/kWhth

        # set the units of an air source heat pump:
        units = {
            "capacity": "kW_th",
            "conversion_factor_electricity": "kWh_el/kWh_th",
            "electricity_input": "kWh_el",
            "heat_output": "kWh_th",
            "COP": "-",
        }
        self.units = OrderedDict(units)
