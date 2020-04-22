def onshore_wind_from_era5(placements, era5_path, gwa_path, esa_cci_path):
    wf = WorkflowGenerator(
        placements
    ).for_onshore_wind_energy(
    ).with_source(
        "ERA5",
        era5_path
    ).read(
        "windspeed"
    ).read(
        "air_temp"
    ).read(
        "pressure"
    ).apply_long_run_average(
        "windspeed",
        source_lra=rk.weather.sources.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_lra=gwa_path
    ).estimate_roughness_from_land_cover(
        path=esa_cci_path,
        source_type="esa"
    ).project_wind_speeds_to_hub_height_with_log_law(
    ).apply_air_density_correction_to_wind_speeds(
    ).convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1
    ).simulate_onshore_wind_energy(
    ).apply_loss_factor(
        # dampens lower wind speeds
        loss=lambda capfacval: (1 - np.exp(5 * capfacval))
    )

    onshore_capacity_factor = wf.sim_data['capacity_factor']

    return onshore_capacity_factor


def offshore_wind_from_era5(placements, era5_path, gwa_path, clc_path):
    """Corresponds to Ryberg et al 2019"""
    wf = WorkflowGenerator(placements)
    wf.for_onshore_wind_energy()
    wf.with_source(
        "ERA5",
        era5_path)
    wf.read("windspeed")
    wf.set_roughness(0.002)
    wf.project_wind_speeds_to_hub_height_with_log_law()
    wf.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1)
    wf.simulate_onshore_wind_energy()
    wf.apply_loss_factor(lambda capfacval: (1 - np.exp(5 * capfacval)))

    onshore_capacity_factor = wf.sim_data['capacity_factor']

    return onshore_capacity_factor


def onshore_wind_from_era5(placements, era5_path, gwa_path, clc_path):
    """Corresponds to Ryberg et al 2019"""
    wf = WorkflowGenerator(placements)
    wf.for_onshore_wind_energy()
    wf.with_source(
        "ERA5",
        era5_path)
    wf.read("windspeed")
    wf.read("air_temp")
    wf.read("pressure")
    wf.apply_long_run_average(
        "windspeed",
        source_lra=rk.weather.sources.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_lra=gwa_path)
    wf.estimate_roughness_from_land_cover(
        path=clc_path,
        source_type="clc")
    wf.project_wind_speeds_to_hub_height_with_log_law()
    wf.apply_air_density_correction_to_wind_speeds()
    wf.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1)
    wf.simulate_onshore_wind_energy()
    wf.apply_loss_factor(lambda capfacval: (1 - np.exp(5 * capfacval)))

    onshore_capacity_factor = wf.sim_data['capacity_factor']

    return onshore_capacity_factor


def onshore_wind_from_era5(placements, era5_path, gwa_path, esa_cci_path):
    """Corresponds to Ryberg et al 2020"""
    wf = WorkflowGenerator(placements)
    wf.for_onshore_wind_energy()
    wf.with_source(
        "ERA5",
        era5_path)
    wf.read("windspeed")
    wf.read("air_temp")
    wf.read("pressure")
    wf.apply_long_run_average(
        "windspeed",
        source_lra=rk.weather.sources.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_lra=gwa_path)
    wf.estimate_roughness_from_land_cover(
        path=esa_cci_path,
        source_type="esa")
    wf.project_wind_speeds_to_hub_height_with_log_law()
    wf.apply_air_density_correction_to_wind_speeds()
    wf.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1)
    wf.simulate_onshore_wind_energy()
    wf.apply_loss_factor(lambda capfacval: (1 - np.exp(5 * capfacval)))

    onshore_capacity_factor = wf.sim_data['capacity_factor']

    return onshore_capacity_factor


def onshore_wind_from_era5_with_wake(placements, era5_path, gwa_path, esa_cci_path):
    """Corresponds to Ryberg et al 2021"""
    wf = WorkflowGenerator(placements)
    wf.for_onshore_wind_energy()
    wf.with_source(
        "ERA5",
        era5_path)
    wf.read(["windspeed", "winddir", "air_temp", "pressure"]).
    wf.apply_long_run_average(
        "windspeed",
        source_lra=rk.weather.sources.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_lra=gwa_path)
    wf.estimate_roughness_from_land_cover(
        path=esa_cci_path,
        source_type="esa")
    wf.project_wind_speeds_to_hub_height_with_log_law()
    wf.apply_air_density_correction_to_wind_speeds()
    wf.apply_wake_effect()
    wf.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1)
    wf.simulate_onshore_wind_energy()
    wf.apply_loss_factor(lambda capfacval: (1 - np.exp(5 * capfacval)))

    onshore_capacity_factor = wf.sim_data['capacity_factor']

    return onshore_capacity_factor


def openfield_pv_fixed_sarah(placements, sarah_path, era5_path, gwa_path, clc_path):
    """Corresponds to Ryberg et al 2019"""
    wf = WorkflowGenerator(placements)
    wf.for_onshore_wind_energy()

    wf.read(
        data=["ghi", 'dni_flat'],
        source_type="SARAH",
        path=sarah_path,
    )

    wf.read(
        data=["windspeed", "air_temp", "pressure"],
        source_type="ERA5",
        path=era5_path,
        time_domain="SARAH"
    )

    wf.apply_long_run_average(
        "windspeed",
        source_lra=rk.weather.sources.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_lra=gwa_path)
    wf.estimate_roughness_from_land_cover(
        path=clc_path,
        source_type="clc")
    wf.project_wind_speeds_to_hub_height_with_log_law()
    wf.apply_air_density_correction_to_wind_speeds()
    wf.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1)
    wf.simulate_onshore_wind_energy()
    wf.apply_loss_factor(lambda capfacval: (1 - np.exp(5 * capfacval)))

    onshore_capacity_factor = wf.sim_data['capacity_factor']

    return onshore_capacity_factor
