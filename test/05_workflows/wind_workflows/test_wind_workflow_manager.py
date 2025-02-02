import pandas as pd
import numpy as np
from reskit.wind import WindWorkflowManager, PowerCurve
import reskit as rk
import pytest
from reskit import TEST_DATA


def test_WindWorkflowManager___init__():
    placements = pd.DataFrame()
    placements["lon"] = [
        6.083,
        6.183,
        6.083,
        6.183,
        6.083,
    ]
    placements["lat"] = [
        50.475,
        50.575,
        50.675,
        50.775,
        50.875,
    ]
    placements["hub_height"] = [
        140,
        140,
        140,
        140,
        140,
    ]
    placements["capacity"] = [2000, 3000, 4000, 5000, 6000]
    placements["rotor_diam"] = [136, 136, 136, 136, 136]

    man = WindWorkflowManager(
        placements, synthetic_power_curve_cut_out=25, synthetic_power_curve_rounding=1
    )

    assert np.isclose(man.ext.xMin, 6.083000)
    assert np.isclose(man.ext.xMax, 6.183000)
    assert np.isclose(man.ext.yMin, 50.475000)
    assert np.isclose(man.ext.yMax, 50.875000)

    assert "SPC:138,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:138,25"], PowerCurve
    )
    assert "SPC:207,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:207,25"], PowerCurve
    )
    assert "SPC:275,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:275,25"], PowerCurve
    )
    assert "SPC:344,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:344,25"], PowerCurve
    )
    assert "SPC:413,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:413,25"], PowerCurve
    )

    assert np.isclose(
        man.powerCurveLibrary["SPC:138,25"].capacity_factor.mean(), 0.5743801652892562
    )
    assert np.isclose(
        man.powerCurveLibrary["SPC:138,25"].capacity_factor.std(), 0.3267113684649957
    )

    assert (man.placements["lon"] == placements["lon"]).all()
    assert (man.placements["lat"] == placements["lat"]).all()
    assert (man.placements["hub_height"] == placements["hub_height"]).all()
    assert (man.placements["capacity"] == placements["capacity"]).all()
    assert (man.placements["rotor_diam"] == placements["rotor_diam"]).all()
    assert (
        man.placements["powerCurve"]
        == ["SPC:138,25", "SPC:207,25", "SPC:275,25", "SPC:344,25", "SPC:413,25"]
    ).all()

    return man


def test_WindWorkflowManager_with_ws___init__():
    man = test_WindWorkflowManager___init__()
    # generate random wind data for only 24 hrs and add to manager
    np.random.seed(seed=12345)
    wind_speeds = np.random.randint(0, 16, size=(24, len(man.placements)))
    man.sim_data["elevated_wind_speed"] = wind_speeds
    # add wind speed height
    man.elevated_wind_speed_height = np.random.randint(
        60, 140, size=(1, len(man.placements))
    )
    return man


@pytest.fixture
def pt_WindWorkflowManager_initialized() -> WindWorkflowManager:
    return test_WindWorkflowManager___init__()


def test_WindWorkflowManager_set_roughness(pt_WindWorkflowManager_initialized):
    man = pt_WindWorkflowManager_initialized

    roughnesses = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    man.set_roughness(roughnesses)
    assert (man.placements["roughness"] == roughnesses).all()


def test_WindWorkflowManager_estimate_roughness_from_land_cover(
    pt_WindWorkflowManager_initialized,
):
    man = pt_WindWorkflowManager_initialized
    man.estimate_roughness_from_land_cover(
        rk.TEST_DATA["clc-aachen_clipped.tif"], source_type="clc"
    )
    assert (man.placements["roughness"] == [0.5, 0.0005, 0.03, 0.03, 0.3]).all()


@pytest.fixture
def pt_WindWorkflowManager_loaded(
    pt_WindWorkflowManager_initialized: WindWorkflowManager,
) -> WindWorkflowManager:
    man = pt_WindWorkflowManager_initialized

    man.read(
        variables=[
            "elevated_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=rk.TEST_DATA["era5-like"],
        set_time_index=True,
        verbose=False,
    )

    return man


def test_WindWorkflowManager_logarithmic_projection_of_wind_speeds_to_hub_height(
    pt_WindWorkflowManager_loaded,
):
    man = pt_WindWorkflowManager_loaded

    roughnesses = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    man.set_roughness(roughnesses)

    man.logarithmic_projection_of_wind_speeds_to_hub_height()

    assert np.isclose(man.sim_data["elevated_wind_speed"].mean(), 8.019319426456706)
    assert np.isclose(man.sim_data["elevated_wind_speed"].std(), 2.9112541058945705)


def test_WindWorkflowManager_apply_air_density_correction_to_wind_speeds(
    pt_WindWorkflowManager_loaded,
):
    man = pt_WindWorkflowManager_loaded

    man.apply_air_density_correction_to_wind_speeds()

    assert np.isclose(man.sim_data["elevated_wind_speed"].mean(), 7.8090427455556375)
    assert np.isclose(man.sim_data["elevated_wind_speed"].std(), 2.822941278260297)


def test_WindWorkflowManager_convolute_power_curves(pt_WindWorkflowManager_initialized):
    man = pt_WindWorkflowManager_initialized
    man.convolute_power_curves(scaling=0.06, base=0.1)

    assert np.isclose(
        man.powerCurveLibrary["SPC:138,25"].capacity_factor.mean(), 0.47633148562562416
    )
    assert np.isclose(
        man.powerCurveLibrary["SPC:138,25"].capacity_factor.std(), 0.4521186399908671
    )


def test_WindWorkflowManager_apply_wake_correction_of_wind_speeds():

    # first without any "wake_curve"
    man = test_WindWorkflowManager_with_ws___init__()
    assert not "wake_curve" in man.placements.columns
    man.apply_wake_correction_of_wind_speeds(wake_curve=None)

    assert np.isclose(
        man.sim_data["elevated_wind_speed"].mean(axis=0),
        np.array([7.29166667, 8.04166667, 8.66666667, 6.33333333, 7.04166667]),
    ).all()

    # now with scalar "wake_curve" function arg
    man = test_WindWorkflowManager_with_ws___init__()
    assert not "wake_curve" in man.placements.columns
    man.apply_wake_correction_of_wind_speeds(wake_curve="dena_mean")

    assert np.isclose(
        man.sim_data["elevated_wind_speed"].mean(axis=0),
        np.array([6.375, 7.08333333, 7.70833333, 5.375, 6.125]),
    ).all()

    # and last with location-specific column value
    man = test_WindWorkflowManager_with_ws___init__()
    man.placements["wake_curve"] = None
    man.placements.loc[2, "wake_curve"] = "dena_mean"
    man.apply_wake_correction_of_wind_speeds(wake_curve=None)

    assert np.isclose(
        man.sim_data["elevated_wind_speed"].mean(axis=0),
        np.array([7.29166667, 8.04166667, 7.70833333, 6.33333333, 7.04166667]),
    ).all()


def test_WindWorkflowManager_simulate(pt_WindWorkflowManager_loaded):
    man = pt_WindWorkflowManager_loaded

    man.simulate()
    assert np.isclose(man.sim_data["capacity_factor"].mean(), 0.4845642857142858)
    assert np.isclose(man.sim_data["capacity_factor"].std(), 0.3275352284371056)

    # test with max_batch_size = 3
    man_batch = pt_WindWorkflowManager_loaded

    man_batch.simulate(max_batch_size=3)
    assert np.isclose(
        man_batch.sim_data["capacity_factor"].mean(),
        man.sim_data["capacity_factor"].mean(),
    )
    assert np.isclose(
        man_batch.sim_data["capacity_factor"].std(),
        man.sim_data["capacity_factor"].std(),
    )

    # check again with scalar powercurve correction
    correct_to = 0.5
    tolerance = 0.05
    man.simulate(cf_correction_factor=correct_to, tolerance=tolerance)

    assert np.isclose(
        man.sim_data["capacity_factor"].mean(),
        0.4845866909936545 * correct_to,
        rtol=tolerance,
    )

    # repeat with correction factor raster
    correction_raster = TEST_DATA[
        "dummy_correction_factors.tif"
    ]  # abuse GSA raster for correction (mean ~2.9)
    man.simulate(cf_correction_factor=correction_raster, tolerance=tolerance)

    avg_corr_factor = 0.8348340150085444  # from dummy data
    assert np.isclose(
        man.sim_data["capacity_factor"].mean(),
        0.4845866909936545 * avg_corr_factor,
        rtol=tolerance,
    )


def test_WindWorkflowManager_mixed_values___init___():
    placements = pd.DataFrame()
    placements["lon"] = [
        6.083,
        6.183,
        6.083,
        6.183,
        6.083,
    ]
    placements["lat"] = [
        50.475,
        50.575,
        50.675,
        50.775,
        50.875,
    ]
    placements["hub_height"] = [
        140,
        140,
        140,
        140,
        140,
    ]
    placements["capacity"] = [2000, 3000, 4000, 3600, 6000]
    placements["rotor_diam"] = [136, 136, 136, 116, 136]
    placements["powerCurve"] = [None, None, None, "V117-3300", None]

    man = WindWorkflowManager(
        placements, synthetic_power_curve_cut_out=25, synthetic_power_curve_rounding=1
    )

    assert "SPC:138,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:138,25"], PowerCurve
    )
    assert "SPC:207,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:207,25"], PowerCurve
    )
    assert "SPC:275,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:275,25"], PowerCurve
    )
    assert "V117-3300" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["V117-3300"], PowerCurve
    )
    assert "SPC:413,25" in man.powerCurveLibrary and isinstance(
        man.powerCurveLibrary["SPC:413,25"], PowerCurve
    )

    assert (man.placements["lon"] == placements["lon"]).all()
    assert (man.placements["lat"] == placements["lat"]).all()
    assert (man.placements["hub_height"] == placements["hub_height"]).all()
    assert (man.placements["capacity"] == placements["capacity"]).all()
    assert (man.placements["rotor_diam"] == placements["rotor_diam"]).all()
    assert (
        man.placements["powerCurve"]
        == ["SPC:138,25", "SPC:207,25", "SPC:275,25", "V117-3300", "SPC:413,25"]
    ).all()

    return man
