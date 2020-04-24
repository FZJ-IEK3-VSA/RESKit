import geokit as gk
import pandas as pd
import numpy as np

import reskit as rk
from reskit import windpower
from reskit.workflows.wind import WindWorkflowGenerator

import pytest

merra_path = rk._TEST_DATA_['weather_data']
gwa_50m_path = rk._TEST_DATA_['gwa50-like.tif']
clc2012_path = rk._TEST_DATA_['clc-aachen_clipped.tif']


def test_init(placements):

    wf = WindWorkflowGenerator(placements)

    assert isinstance(wf.locs, gk.LocationSet)
    assert wf.locs.count == placements.shape[0]

    assert len(wf.powerCurveLibrary.keys()) == 2
    assert isinstance(
        wf.powerCurveLibrary['SPC:301,25'], rk.windpower.PowerCurve)
    assert isinstance(
        wf.powerCurveLibrary['SPC:226,25'], rk.windpower.PowerCurve)

    assert np.isclose(
        wf.powerCurveLibrary['SPC:301,25'].cf.mean(), 0.5743801652892562)


@pytest.mark.parametrize('variables, source_type, path, expected',
                         [(['elevated_wind_speed', "surface_pressure", "surface_air_temperature"], "MERRA", merra_path, [50, 8.595122807949584, 2.8790115195804775])])
def test_read(wind_workflow, variables, source_type, path, expected):

    wind_workflow.read(
        variables=variables,
        source_type=source_type,
        path=path,
        set_time_index=True,
        verbose=False
    )

    assert "elevated_wind_speed" in wind_workflow.sim_data
    assert "surface_pressure" in wind_workflow.sim_data
    assert "surface_air_temperature" in wind_workflow.sim_data

    assert wind_workflow.sim_data['elevated_wind_speed'].shape[0] == len(
        wind_workflow.time_index)
    assert wind_workflow.sim_data['elevated_wind_speed'].shape[1] == wind_workflow.locs.count

    assert wind_workflow.elevated_wind_speed_height == expected[0]

    assert np.isclose(
        wind_workflow.sim_data['elevated_wind_speed'].values.mean(), expected[1])
    assert np.isclose(
        wind_workflow.sim_data['elevated_wind_speed'].values.std(), expected[2])


def test_adjust(wind_workflow):

    wind_workflow.read(
        variables=['elevated_wind_speed', "surface_pressure", "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wind_workflow.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    assert np.isclose(wind_workflow.sim_data['elevated_wind_speed'].values.mean(), 6.316895937544227)
    assert np.isclose(wind_workflow.sim_data['elevated_wind_speed'].values.std(), 2.165638796355242)


def test_estimate_roughness(wind_workflow):

    wind_workflow.read(
        variables=['elevated_wind_speed', "surface_pressure", "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wind_workflow.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    wind_workflow.estimate_roughness_from_land_cover(
        path=clc2012_path,
        source_type="clc")

    assert np.isclose(wind_workflow.placements['roughness'].mean(), 0.21235714285714286)
    assert np.isclose(wind_workflow.placements['roughness'].std(), 0.22295718871338902)


def test_logarithmic_projection_of_wind_speeds_to_hub_height(wind_workflow):
    wind_workflow.read(
        variables=['elevated_wind_speed', "surface_pressure", "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wind_workflow.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    wind_workflow.estimate_roughness_from_land_cover(
        path=clc2012_path,
        source_type="clc")

    wind_workflow.logarithmic_projection_of_wind_speeds_to_hub_height()

    assert np.isclose(wind_workflow.sim_data['elevated_wind_speed'].values.mean(), 7.247590523853572)
    assert np.isclose(wind_workflow.sim_data['elevated_wind_speed'].values.std(), 2.4773404233790046)


def test_apply_air_density_correction(wind_workflow):
    wind_workflow.read(
        variables=['elevated_wind_speed', "surface_pressure", "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wind_workflow.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    wind_workflow.estimate_roughness_from_land_cover(
        path=clc2012_path,
        source_type="clc")

    wind_workflow.logarithmic_projection_of_wind_speeds_to_hub_height()

    wind_workflow.apply_air_density_correction_to_wind_speeds()

    assert np.isclose(wind_workflow.sim_data['elevated_wind_speed'].values.mean(), 7.308111114850857)
    assert np.isclose(wind_workflow.sim_data['elevated_wind_speed'].values.std(), 2.4932367249100644)


def test_convolute_power_curves(wind_workflow):
    wind_workflow.read(
        variables=['elevated_wind_speed', "surface_pressure", "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wind_workflow.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    wind_workflow.estimate_roughness_from_land_cover(
        path=clc2012_path,
        source_type="clc")

    wind_workflow.logarithmic_projection_of_wind_speeds_to_hub_height()

    wind_workflow.apply_air_density_correction_to_wind_speeds()

    wind_workflow.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1
    )
    assert np.isclose(wind_workflow.powerCurveLibrary['SPC:301,25'].cf.mean(), 0.42844125885573553)


def test_simulate(wind_workflow):
    wind_workflow.read(
        variables=['elevated_wind_speed', "surface_pressure", "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wind_workflow.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    wind_workflow.estimate_roughness_from_land_cover(
        path=clc2012_path,
        source_type="clc")

    wind_workflow.logarithmic_projection_of_wind_speeds_to_hub_height()

    wind_workflow.apply_air_density_correction_to_wind_speeds()

    wind_workflow.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1
    )

    wind_workflow.simulate()

    assert np.isclose(wind_workflow.sim_data['capacity_factor'].values.mean(), 0.44716889827228373)
    assert np.isclose(wind_workflow.sim_data['capacity_factor'].values.std(), 0.3191091435347311)


def test_apply_loss_factor(wind_workflow):

    wind_workflow.read(
        variables=['elevated_wind_speed', "surface_pressure", "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wind_workflow.adjust_variable_to_long_run_average(
        variable='elevated_wind_speed',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    wind_workflow.estimate_roughness_from_land_cover(
        path=clc2012_path,
        source_type="clc")

    wind_workflow.logarithmic_projection_of_wind_speeds_to_hub_height()

    wind_workflow.apply_air_density_correction_to_wind_speeds()

    wind_workflow.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1
    )

    wind_workflow.simulate()

    wind_workflow.apply_loss_factor(loss=lambda x: windpower.lowGenCorrection(x, base=0.0, sharpness=5.0))

    # tests
    assert np.isclose(wind_workflow.sim_data['capacity_factor'].values.mean(), 0.4045317473094216)
    assert np.isclose(wind_workflow.sim_data['capacity_factor'].values.std(), 0.3383322799886796)
