from ... import weather as rk_weather
from ... import util as rk_util
from .wind_workflow_manager import WindWorkflowManager


def onshore_wind_merra_ryberg2019_europe(placements, merra_path, gwa_50m_path, clc2012_path, output_netcdf_path=None, output_variables=None):
    # TODO: Add range limitation over Europe by checking placements
    """
    Simulates onshore wind generation in Europe using NASA's MERRA2 database [1]

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    merra_path : str
        Path to the MERRA2 data. See ___ for data processing requirements.
    gwa_50m_path : str
        Path to the Global Wind Atlas at 50m [2] rater file.
    clc2012_path : str
        Path to the CLC 2012 raster file [3].
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Additional variables?, by default None

    Returns
    -------
    xarrax
        A xarray dataset including all the output variables you defined as your output_variables.??

    Sources
    ------
    [1] NASA (National Aeronautics and Space Administration). (2019). Modern-Era Retrospective analysis for Research and Applications, Version 2. NASA Goddard Earth Sciences (GES) Data and Information Services Center (DISC). https://disc.gsfc.nasa.gov/datasets?keywords=%22MERRA-2%22&page=1&source=Models%2FAnalyses MERRA-2
    [2] DTU Wind Energy. (2019). Gloabal Wind Atlas. https://globalwindatlas.info/
    [3] Copernicus (European Union’s Earth Observation Programme). (2012). Corine Land Cover 2012. Copernicus. https://land.copernicus.eu/pan-european/corine-land-cover/clc-2012
    
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
    
    """
    Simulates offshore wind generation using NASA's MERRA2 database [1].

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    merra_path : str
        Path to the MERRA2 data. See ___ for data processing requirements.
    output_netcdf_path : str, optional
        Path to a durectory to put the output files, by default None
    output_variables : str, optional
        Additional variables?, by default None

    Returns
    -------
    xarrax
        ###Check Workflow.py

    Sources
    ------
    [1] National Aeronautics and Space Administration. (2019). Modern-Era Retrospective analysis for Research and Applications, Version 2. NASA Goddard Earth Sciences (GES) Data and Information Services Center (DISC). https://disc.gsfc.nasa.gov/datasets?keywords=%22MERRA-2%22&page=1&source=Models%2FAnalyses MERRA-2

    """    
    
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
    """
    Simulates offshore wind generation using NASA's ERA5 database [1].

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    era5_path : str
        Path to the ERA5 data. See ___ for data processing requirements.
    output_netcdf_path : str, optional
        Path to a directory to put the output files, by default None
    output_variables : str, optional
        Additional variables?, by default None

    Returns
    -------
    xarrax
        ###Check Workflow.py

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5


    """    
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
    """
    Simulates onshore wind generation using ECMWF's ERA5 database [1]

    Parameters
    ----------
    placements : pandas Dataframe
        A Dataframe object with the parameters needed by the simulation.
    era5_path : str
        Path to the ERA5 data. See ___ for data processing requirements.
    gwa_100m_path : str
        Path to the Global Wind Atlas at 100m [2] rater file.
    esa_cci_path : str
        Path to the ESA CCI raster file [3].
    output_netcdf_path : str, optional
        Path to a durectory to put the output files, by default None
    output_variables : str, optional
        Additional variables?, by default None

    Returns
    -------
    xarrax
        ###Check Workflow.py

    Sources
    ------
    [1] European Centre for Medium-Range Weather Forecasts. (2019). ERA5 dataset. https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
    [2] DTU Wind Energy. (2019). Gloabal Wind Atlas. https://globalwindatlas.info/
    [3] ESA. Land Cover CCI Product User Guide Version 2. Tech. Rep. (2017). Available at: maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf
    """
    
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
