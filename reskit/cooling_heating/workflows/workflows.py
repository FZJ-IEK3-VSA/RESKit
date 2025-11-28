# import primary packages
from typing import List

import numpy as np
import pandas as pd

# import othert modules
from .cooling_heating_workflow_manager import CoolingHeatingWorkflowManager


def air_cooling_wenzel2025(
    placements: pd.DataFrame,
    era5_path: str,
    temperatureCoolant: int | float,
    designTemperature: int | float,
    heatTransferDelta: int | float = 5,
    efficiencyFan: int | float = 0.7,
    pressureDropAir: int | float = 261,
    efficiencyPump: int | float = 0.7,
    pressureDropWater: int | float = 200000,
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
    assert isinstance(temperatureCoolant, (int, float))
    assert isinstance(designTemperature, (int, float))
    assert isinstance(heatTransferDelta, (int, float))
    assert isinstance(efficiencyFan, (int, float))
    assert isinstance(pressureDropAir, (int, float))
    assert isinstance(efficiencyPump, (int, float))
    assert isinstance(pressureDropWater, (int, float))
    assert 0 < efficiencyFan <= 1, "efficiencyFan must be between 0 and 1"
    assert 0 < efficiencyPump <= 1, "efficiencyPump must be between 0 and 1"

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
        temperatureCoolant,
        heatTransferDelta=heatTransferDelta,
        efficiencyFan=efficiencyFan,
        pressureDropAir=pressureDropAir,
        designTemperature=None,
    )
    wf.calculate_pump_power_air_cooling(
        temperatureCoolant,
        heatTransferDelta=heatTransferDelta,
        efficiencyPump=efficiencyPump,
        pressureDropWater=pressureDropWater,
        designTemperature=None,
    )
    wf.calculate_relative_cost_factor_air_cooling(
        designTemperature,
        temperatureCoolant,
        heatTransferDelta=heatTransferDelta,
        efficiencyFan=efficiencyFan,
        efficiencyPump=efficiencyPump,
        pressureDropAir=pressureDropAir,
        pressureDropWater=pressureDropWater,
    )

    wf.calculate_capacity_factor_air_cooling(
        designTemperature=designTemperature,
        temperatureCoolant=temperatureCoolant,
        heatTransferDelta=heatTransferDelta,
        efficiencyFan=efficiencyFan,
        efficiencyPump=efficiencyPump,
        pressureDropAir=pressureDropAir,
        pressureDropWater=pressureDropWater,
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
    targetTemperature: int | float = 100,
    secondLawEfficiency: int | float = 0.5,
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
    assert isinstance(targetTemperature, (int, float))
    assert isinstance(secondLawEfficiency, (int, float))
    assert 0 < secondLawEfficiency <= 1, "efficiency must be between 0 and 1"

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
    wf.simulate_air_source_heat_pump(targetTemperature=targetTemperature, secondLawEfficiency=secondLawEfficiency)

    wf.sim_data["electricity_input"] = -wf.sim_data["conversion_factor_electricity"] * np.array(
        wf.placements["capacity"]
    )  # kWh_el/h

    wf.sim_data["heat_output"] = np.ones(wf.sim_data["electricity_input"].shape) * np.array(wf.placements["capacity"])

    return wf.to_xarray(
        output_netcdf_path=output_netcdf_path,
        output_variables=output_variables,
        custom_attributes=wf.units,
    )
