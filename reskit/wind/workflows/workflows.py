from ... import weather as rk_weather
from ... import util as rk_util
from .wind_workflow_manager import WindWorkflowManager


def onshore_wind_merra_ryberg2019_europe(placements, merra_path, gwa_50m_path, clc2012_path, output_netcdf_path=None, output_variables=None):
    # TODO: Add range limitation over Europe by checking placements
    """
    [summary]

    Parameters
    ----------
    placements : [type]
        [description]
    merra_path : [type]
        [description]
    gwa_50m_path : [type]
        [description]
    clc2012_path : [type]
        [description]
    output_netcdf_path : [type], optional
        [description], by default None
    output_variables : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    
    wf = WindWorkflowManager(placements)

    wf.read(
        variables=['elevated_wind_speed',
                   "surface_pressure",
                   "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False)

    wf.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk_weather.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    wf.estimate_roughness_from_land_cover(
        path=clc2012_path,
        source_type="clc")

    wf.logarithmic_projection_of_wind_speeds_to_hub_height()

    wf.apply_air_density_correction_to_wind_speeds()

    wf.convolute_power_curves(
        scaling=0.06,
        base=0.1
    )

    wf.simulate()

    wf.apply_loss_factor(
        loss=lambda x: rk_util.low_generation_loss(x, base=0.0, sharpness=5.0)
    )

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def offshore_wind_merra_caglayan2019(placements, merra_path, output_netcdf_path=None, output_variables=None):
    wf = WindWorkflowManager(placements)

    wf.read(
        variables=['elevated_wind_speed', ],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False)

    wf.set_roughness(0.0002)

    wf.logarithmic_projection_of_wind_speeds_to_hub_height()

    wf.convolute_power_curves(
        scaling=0.04,  # TODO: Check values with Dil
        base=0.5       # TODO: Check values with Dil
    )

    wf.simulate()

    wf.apply_loss_factor(
        loss=lambda x: rk_util.low_generation_loss(x, base=0.1, sharpness=3.5)  # TODO: Check values with Dil
    )

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def offshore_wind_era5_unvalidated(placements, era5_path, output_netcdf_path=None, output_variables=None):
    """[] """
    wf = WindWorkflowManager(placements)

    wf.read(
        variables=['elevated_wind_speed', ],
        source_type="ERA5",
        path=era5_path,
        set_time_index=True,
        verbose=False)

    wf.set_roughness(0.0002)

    wf.logarithmic_projection_of_wind_speeds_to_hub_height()

    wf.convolute_power_curves(
        scaling=0.04,  # TODO: Check values with Dil
        base=0.5       # TODO: Check values with Dil
    )

    wf.simulate()

    wf.apply_loss_factor(
        loss=lambda x: rk_util.low_generation_loss(x, base=0.1, sharpness=3.5)  # TODO: Check values with Dil
    )

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def onshore_wind_era5_unvalidated(placements, era5_path, gwa_100m_path, esa_cci_path, output_netcdf_path=None, output_variables=None):
    wf = WindWorkflowManager(placements)

    wf.read(
        variables=['elevated_wind_speed',
                   "surface_pressure",
                   "surface_air_temperature"],
        source_type="ERA5",
        path=era5_path,
        set_time_index=True,
        verbose=False)

    wf.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_100m_path
    )

    wf.estimate_roughness_from_land_cover(
        path=esa_cci_path,
        source_type="cci")

    wf.logarithmic_projection_of_wind_speeds_to_hub_height()

    wf.apply_air_density_correction_to_wind_speeds()

    wf.convolute_power_curves(
        scaling=0.06,
        base=0.1
    )

    wf.simulate()

    wf.apply_loss_factor(
        loss=lambda x: rk_util.low_generation_loss(x, base=0.0, sharpness=5.0)
    )

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)
