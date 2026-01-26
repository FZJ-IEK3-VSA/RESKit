# import primary packages
from typing import List

import numpy as np
import pandas as pd

# import othert modules
from reskit.cooling_heating.workflows.cooling_heating_workflow_manager import CoolingHeatingWorkflowManager
from reskit.util.relative_humidity import calculate_relative_humidity
from reskit.util.wet_bulb_temperature import calculate_wet_bulb_temperature


def evaporative_cooling_wortmann2025(
    placements: pd.DataFrame,
    era5_path: str,
    temperature_coolant: int | float,
    design_temperature: int | float,
    heat_transfer_delta: int | float = 5,
    efficiency_dan: int | float = 0.7,
    pressure_drop_air: int | float = 261,
    efficiency_pump: int | float = 0.7,
    pressure_drop_water: int | float = 200000,
    output_netcdf_path: str = None,
    output_variables: List[str] = None,
):
    """
    Simulate an evaporative-cooling system based on ERA5 weather data.

    This function calculates the fan and pump power requirements, capacity factor,
    and total electricity demand for air-cooling systems at varying ambient temperatures.
    Results can be saved to a NetCDF file.

    Parameters
    ----------
    placements : pd.DataFrame
        DataFrame specifying the plant locations and their capacities.
    era5_path : str
        Path to the ERA5 weather data source.
    temperature_coolant : float
        Temperature of the heat load to be cooled [°C].
    design_temperature : float
        Temperature for the nominal design point of the air cooling system [°C].
    heat_transfer_delta : float, optional
        Temperature difference required for heat transfer from air to coolant [K]. Default is 5.
    efficiency_dan : float, optional
        Efficiency of the fan system [0, 1]. Default is 0.7.
    pressure_drop_air : float, optional
        Pressure drop of air through the cooling frame channels [Pa]. Default is 261.
    efficiency_pump : float, optional
        Efficiency of the pump system [0, 1]. Default is 0.7.
    pressure_drop_water : float, optional
        Pressure drop of water through the circuit [Pa]. Default is 200000.
    output_netcdf_path : str, optional
        Path to save the output NetCDF file. Default is None.
    output_variables : list of str, optional
        List of simulation variables to save to the NetCDF file. If None, all variables are saved.

    Returns
    -------
    xarray.Dataset
        Simulation results, including capacity factor, fan/pump/electricity inputs, and cooling output.
        Can be limited to `output_variables` if specified.

    Notes
    -----
    - Calculates fan and pump power using `calculate_fan_power_air_cooling` and `calculate_pump_power_air_cooling`.
    - Computes the system capacity factor relative to the design temperature using `calculate_capacity_factor_air_cooling`.
    - Total electricity input includes contributions from both fan and pump.
    - Stores all relevant units in `wf.units` for reference.

    Raises
    ------
    AssertionError
        If input parameters are not of expected type or if efficiency values are not within (0, 1].
    """
    assert isinstance(temperature_coolant, (int, float))
    assert isinstance(design_temperature, (int, float))
    assert isinstance(heat_transfer_delta, (int, float))
    assert isinstance(efficiency_dan, (int, float))
    assert isinstance(pressure_drop_air, (int, float))
    assert isinstance(efficiency_pump, (int, float))
    assert isinstance(pressure_drop_water, (int, float))
    assert 0 < efficiency_dan <= 1, "efficiencyFan must be between 0 and 1"
    assert 0 < efficiency_pump <= 1, "efficiencyPump must be between 0 and 1"

    wf = CoolingHeatingWorkflowManager(placements)


def air_cooling_wenzel2025(
    placements: pd.DataFrame,
    era5_path: str,
    temperature_coolant: int | float,
    design_temperature: int | float,
    heat_transfer_delta: int | float = 5,
    efficiency_fan: int | float = 0.7,
    pressure_drop_air: int | float = 261,
    efficiency_pump: int | float = 0.7,
    pressure_drop_water: int | float = 200000,
    output_netcdf_path: str = None,
    output_variables: List[str] = None,
):
    """
    Simulate an air-cooling system based on ERA5 weather data.

    This function calculates the fan and pump power requirements, capacity factor,
    and total electricity demand for air-cooling systems at varying ambient temperatures.
    Results can be saved to a NetCDF file.

    Parameters
    ----------
    placements : pd.DataFrame
        DataFrame specifying the plant locations and their capacities.
    era5_path : str
        Path to the ERA5 weather data source.
    temperatureCoolant : float
        Temperature of the heat load to be cooled [°C].
    designTemperature : float
        Temperature for the nominal design point of the air cooling system [°C].
    heatTransferDelta : float, optional
        Temperature difference required for heat transfer from air to coolant [K]. Default is 5.
    efficiencyFan : float, optional
        Efficiency of the fan system [0, 1]. Default is 0.7.
    pressureDropAir : float, optional
        Pressure drop of air through the cooling frame channels [Pa]. Default is 261.
    efficiencyPump : float, optional
        Efficiency of the pump system [0, 1]. Default is 0.7.
    pressureDropWater : float, optional
        Pressure drop of water through the circuit [Pa]. Default is 200000.
    output_netcdf_path : str, optional
        Path to save the output NetCDF file. Default is None.
    output_variables : list of str, optional
        List of simulation variables to save to the NetCDF file. If None, all variables are saved.

    Returns
    -------
    xarray.Dataset
        Simulation results, including capacity factor, fan/pump/electricity inputs, and cooling output.
        Can be limited to `output_variables` if specified.

    Notes
    -----
    - Calculates fan and pump power using `calculate_fan_power_air_cooling` and `calculate_pump_power_air_cooling`.
    - Computes the system capacity factor relative to the design temperature using `calculate_capacity_factor_air_cooling`.
    - Total electricity input includes contributions from both fan and pump.
    - Stores all relevant units in `wf.units` for reference.

    Raises
    ------
    AssertionError
        If input parameters are not of expected type or if efficiency values are not within (0, 1].
    """
    assert isinstance(temperature_coolant, (int, float))
    assert isinstance(design_temperature, (int, float))
    assert isinstance(heat_transfer_delta, (int, float))
    assert isinstance(efficiency_fan, (int, float))
    assert isinstance(pressure_drop_air, (int, float))
    assert isinstance(efficiency_pump, (int, float))
    assert isinstance(pressure_drop_water, (int, float))
    assert 0 < efficiency_fan <= 1, "efficiencyFan must be between 0 and 1"
    assert 0 < efficiency_pump <= 1, "efficiencyPump must be between 0 and 1"

    wf = CoolingHeatingWorkflowManager(placements)

    wf.read(
        variables=[
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )

    wf.calculate_fan_power_air_cooling(
        temperature_coolant,
        heat_transfer_delta=heat_transfer_delta,
        efficiency_fan=efficiency_fan,
        pressure_drop_air=pressure_drop_air,
        design_temperature=None,
    )
    wf.calculate_pump_power_air_cooling(
        temperature_coolant,
        heat_transfer_delta=heat_transfer_delta,
        efficiency_pump=efficiency_pump,
        pressure_drop_water=pressure_drop_water,
        design_temperature=None,
    )
    wf.calculate_relative_cost_factor_air_cooling(
        design_temperature,
        temperature_coolant,
        heat_transfer_delta=heat_transfer_delta,
        efficiency_fan=efficiency_fan,
        efficiency_pump=efficiency_pump,
        pressure_drop_air=pressure_drop_air,
        pressure_drop_water=pressure_drop_water,
    )

    wf.calculate_capacity_factor_air_cooling(
        design_temperature=design_temperature,
        temperature_coolant=temperature_coolant,
        heat_transfer_delta=heat_transfer_delta,
        efficiency_fan=efficiency_fan,
        efficiency_pump=efficiency_pump,
        pressure_drop_air=pressure_drop_air,
        pressure_drop_water=pressure_drop_water,
    )

    # calculate total conversion factor electricity:
    wf.sim_data["conversion_factor_electricity"] = (
        wf.sim_data["conversion_factor_fan_electricity"] + wf.sim_data["conversion_factor_pump_electricity"]
    )  # kWh_el/kWh_th

    # Calculate needed electricity in each time step for design capacity:
    wf.sim_data["cooling_output"] = wf.sim_data["capacity_factor"] * np.array(wf.placements["capacity"])  # kWh_th/h
    wf.sim_data["electricity_input"] = (
        -wf.sim_data["conversion_factor_electricity"]
        * wf.sim_data["capacity_factor"]
        * np.array(wf.placements["capacity"])
    )  # kWh_el/h
    wf.sim_data["electricity_input_fan"] = (
        -wf.sim_data["conversion_factor_fan_electricity"]
        * wf.sim_data["capacity_factor"]
        * np.array(wf.placements["capacity"])
    )  # kWh_el/h
    wf.sim_data["electricity_input_pump"] = (
        -wf.sim_data["conversion_factor_pump_electricity"]
        * wf.sim_data["capacity_factor"]
        * np.array(wf.placements["capacity"])
    )  # kWh_el/h

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
        custom_attributes=wf.units,
    )


def air_source_heat_pump(
    placements: pd.DataFrame,
    era5_path: str,
    target_temperature: int | float = 100,
    second_law_efficiency: int | float = 0.5,
    output_netcdf_path: str = None,
    output_variables: List[str] = None,
):
    """
    Simulate an air-source heat pump based on ERA5 weather data.

    This function calculates the coefficient of performance (COP) and electricity
    demand of an air-source heat pump at varying ambient temperatures. Results
    can be saved to a NetCDF file.

    Parameters
    ----------
    placements : pd.DataFrame
        DataFrame specifying plant locations and capacities.
    era5_path : str
        Path to the ERA5 weather data source.
    targetTemperature : float, optional
        Temperature at which the heat pump should supply the heat [°C]. Default is 100.
    secondLawEfficiency : float, optional
        Second law efficiency of the heat pump [0,1]. Default is 0.5.
    output_netcdf_path : str, optional
        Path to save the output NetCDF file. Default is None.
    output_variables : list of str, optional
        List of simulation variables to save to the NetCDF file. If None, all variables are saved.

    Returns
    -------
    xarray.Dataset
        Simulation results including COP, electricity conversion factor, and electricity input.
        Can be limited to `output_variables` if specified.

    Raises
    ------
    AssertionError
        If `targetTemperature` or `secondLawEfficiency` are not numeric or if
        `secondLawEfficiency` is not within (0, 1].

    Notes
    -----
    - The electricity conversion factor is calculated as -1/COP [kWh_el/kWh_th].
    - Electricity input is computed for each plant based on its capacity and the COP.
    - Units are stored in `wf.units` for reference.
    """
    assert isinstance(target_temperature, (int, float))
    assert isinstance(second_law_efficiency, (int, float))
    assert 0 < second_law_efficiency <= 1, "efficiency must be between 0 and 1"

    wf = CoolingHeatingWorkflowManager(placements)
    wf.read(
        variables=[
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )
    wf.simulate_air_source_heat_pump(target_temperature=target_temperature, second_law_efficiency=second_law_efficiency)

    wf.sim_data["electricity_input"] = -wf.sim_data["conversion_factor_electricity"] * np.array(
        wf.placements["capacity"]
    )  # kWh_el/h

    wf.sim_data["heat_output"] = np.ones(wf.sim_data["electricity_input"].shape) * np.array(wf.placements["capacity"])

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
        custom_attributes=wf.units,
    )


def evaporative_cooling_wortmann2025(
    placements: pd.DataFrame,
    era5_path: str,
    temperature_coolant: int | float,
    heat_transfer_delta: int | float,
    efficiency_cooling_tower: int | float,
    factor_drift_losses: float | int = 0.001,
    typical_cycles_blow_down: int = 5,
    output_netcdf_path: str = None,
    output_variables: List[str] = None,
):
    """
    Simulate an evaporative-cooling system based on ERA5 weather data.

    This function calculates the water losses of an evaporative-cooling systems at varying ambient conditions (temperature, humidity).
    Results can be saved to a NetCDF file.

    Parameters
    ----------
    placements : pd.DataFrame
        DataFrame specifying the plant locations and their capacities.
    era5_path : str
        Path to the ERA5 weather data source.
    temperatureCoolant : float | int
        Temperature of the cooling load (lower temperature if sensible heat transfer) in °C.
    heatTransferDelta : float | int
        Temperature difference required for heat transfer from air to coolant [K]
    efficiencyCoolingTower : float | int
        Efficiency of the cooling tower system [0, 1]
    factorDriftLosses : float | int
        Drift losses by small water droplets carried away by the exhaust air. Defaults to 0.001.
    typical_cycles_blowdown: int
        after how many cycles the blowdown occurs to prevent accumulation of impurities. Defaults to 5.
    output_netcdf_path : str, optional
        Path to save the output NetCDF file. Default is None.
    output_variables : list of str, optional
        List of simulation variables to save to the NetCDF file. If None, all variables are saved.

    Returns
    -------
    xarray.Dataset
        Simulation results, including water losses.
        Can be limited to `output_variables` if specified.

    Notes
    -----
    - Stores all relevant units in `wf.units` for reference.

    Raises
    ------
    AssertionError
        If input parameters are not of expected type or if efficiency values are not within (0, 1].
    """
    assert isinstance(temperature_coolant, (int, float))
    assert isinstance(heat_transfer_delta, (int, float))
    assert isinstance(efficiency_cooling_tower, (int, float))
    assert isinstance(factor_drift_losses, (int, float))
    assert isinstance(typical_cycles_blow_down, (int, float))
    assert 0 < efficiency_cooling_tower <= 1, "efficiencyCoolingTower must be between 0 and 1"
    assert 0 < factor_drift_losses <= 1, "factorDriftLosses must be between 0 and 1"

    wf = CoolingHeatingWorkflowManager(placements)

    wf.read(
        variables=["surface_air_temperature", "surface_dew_temperature"],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )

    wf.sim_data["relative_humidity"] = calculate_relative_humidity(
        dewpoint_temperature=wf.sim_data["surface_dew_temperature"],
        air_temperature=wf.sim_data["surface_air_temperature"],
    )

    wf.sim_data["wet_bulb_temperature"] = calculate_wet_bulb_temperature(
        air_temperature=wf.sim_data["surface_air_temperature"], relative_humidity=wf.sim_data["relative_humidity"]
    )

    wf.calculate_approach_evaporative_cooling(
        temperature_coolant=temperature_coolant,
        heat_transfer_delta=heat_transfer_delta,
        efficiency_cooling_tower=efficiency_cooling_tower,
    )

    wf.calculate_water_losses_evaporative_cooling(
        temperature_coolant=temperature_coolant,
        heat_transfer_delta=heat_transfer_delta,
        efficiency_cooling_tower=efficiency_cooling_tower,
        factor_drift_losses=factor_drift_losses,
        typical_cycles_blowdown=typical_cycles_blow_down,
    )

    # calculate total water demand for the plant
    wf.sim_data["total_water_losses"] = -wf.sim_data["conversion_factor_water"] * np.array(wf.placements["capacity"])

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
        custom_attributes=wf.units,
    )
