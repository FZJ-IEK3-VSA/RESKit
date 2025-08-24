# import primary packages
import numpy as np


# import othert modules
from .cooling_heating_workflow_manager import CoolingHeatingWorkflowManager



def air_cooling_wenzel2025(placements, era5_path, temperatureCoolant, designTemperature, heatTransferDelta=5, efficiencyFan=0.7, pressureDropAir=261, efficiencyPump=0.7, pressureDropWater=200000,  output_netcdf_path = None, output_variables=None):
    """
    Simulation of an Air Cooling Systems based on ERA5 weather data.
    
    temperatureCoolant: float
            Temperature of the heat load to be cooled. [°C]
    designTemperature: float
            Temperature for the nominal design point of the air cooling system [°C]
    heatTransferDelta: float
            temperature delta required for heat transfer from air to coolant [K]. 
    efficiencyFan: float
            efficiency of the total fan system [0,1]. 
    pressureDropAir: float 
            pressure drop of the air through the channels of the frame [Pa]. 
    efficiencyPump: float
            efficiency of the total pump system [0,1]. 
    pressureDropWater: float
            pressure drop of the water which is circulated from the site of the heat load to the A-frame [Pa].
    output_netcdf_path: str
            Path to a file that you want to save your output NETCDF file at.
            Default is None
    output_variables: str
            Output variables of the simulation that you want to save into your NETCDF Outputfile.
    """
    assert isinstance(temperatureCoolant,(int, float))
    assert isinstance(designTemperature,(int, float))
    assert isinstance(heatTransferDelta,(int, float))
    assert isinstance(efficiencyFan,(int, float))
    assert isinstance(pressureDropAir,(int, float))
    assert isinstance(efficiencyPump,(int, float))
    assert isinstance(pressureDropWater,(int, float))
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

    wf.calculate_fan_power_air_cooling(temperatureCoolant, heatTransferDelta=heatTransferDelta, efficiencyFan=efficiencyFan, pressureDropAir=pressureDropAir, designTemperature=None)
    wf.calculate_pump_power_air_cooling(temperatureCoolant, heatTransferDelta=heatTransferDelta, efficiencyPump=efficiencyPump, pressureDropWater=pressureDropWater, designTemperature=None)
    wf.calculate_capacity_factor_air_cooling(designTemperature, temperatureCoolant, heatTransferDelta=heatTransferDelta, efficiencyFan=efficiencyFan, efficiencyPump=efficiencyPump, pressureDropAir=pressureDropAir, pressureDropWater=pressureDropWater)

    #calculate total conversion factor electricity:
    wf.sim_data["conversion_factor_electricity"] = wf.sim_data["conversion_factor_fan_electricity"] + wf.sim_data["conversion_factor_pump_electricity"]

    #Calculate needed electricity in each time step for design capacity:
    wf.sim_data["cooling_output"] = wf.sim_data["capacity_factor"] * np.array(wf.placements["capacity"])
    wf.sim_data["electricity_input"] = -wf.sim_data["conversion_factor_electricity"] * wf.sim_data["capacity_factor"] * np.array(wf.placements["capacity"])
    wf.sim_data["electricity_input_fan"] = -wf.sim_data["conversion_factor_fan_electricity"] * wf.sim_data["capacity_factor"] * np.array(wf.placements["capacity"])
    wf.sim_data["electricity_input_pump"] = -wf.sim_data["conversion_factor_pump_electricity"] * wf.sim_data["capacity_factor"] * np.array(wf.placements["capacity"])


    return wf.to_xarray(
                output_netcdf_path=output_netcdf_path, output_variables=output_variables
            )

def air_source_heat_pump(placements, era5_path, targetTemperature=100, secondLawEfficiency=0.5, output_netcdf_path = None, output_variables=None):
    """
    Simulation of an air source heat pump based on ERA5 weather data.
    
    targetTemperature: float
            Temperature at which the heat pump should supply the heat. [°C]
    secondLawEfficiency: float
            Second Law efficiency of the heat pump.
    output_netcdf_path: str
            Path to a file that you want to save your output NETCDF file at.
            Default is None
    output_variables: str
            Output variables of the simulation that you want to save into your NETCDF Outputfile.
    """
    assert isinstance(targetTemperature,(int, float))
    assert isinstance(secondLawEfficiency,(int, float))
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

    wf.sim_data["electricity_input"] = -wf.sim_data["conversion_factor_electricity"] * np.array(wf.placements["capacity"])
    
    return wf.to_xarray(
                output_netcdf_path=output_netcdf_path, output_variables=output_variables
            )