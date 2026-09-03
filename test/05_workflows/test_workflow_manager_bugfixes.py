"""Regression tests for the workflow manager defects of the maintenance audit."""

import numpy as np
import pandas as pd
import pytest
import xarray

from reskit import WorkflowManager, distribute_workflow, load_workflow_result


def _placements(lons, lats):
    return pd.DataFrame({"lon": list(lons), "lat": list(lats)})


def _loaded_manager(lons=(6.0, 6.1), lats=(50.0, 50.1), periods=4, freq="h", start="2020-01-01"):
    """Build a small manager which holds simulation data, without any weather source."""
    placements = _placements(lons, lats)
    man = WorkflowManager(placements)
    man.set_time_index(pd.date_range(start, periods=periods, freq=freq))
    man.sim_data["capacity_factor"] = np.full((periods, len(placements)), 0.5)
    return man



def test_init_rejects_a_batch_with_one_invalid_longitude():
    # one valid coordinate must not make the whole batch valid
    placements = _placements([6.0, 999.0], [50.0, 50.1])

    with pytest.raises(ValueError, match="lon"):
        WorkflowManager(placements)


def test_init_rejects_a_batch_with_one_invalid_latitude():
    placements = _placements([6.0, 6.1], [50.0, -91.0])

    with pytest.raises(ValueError, match="lat"):
        WorkflowManager(placements)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_init_rejects_nan_and_infinite_coordinates(value):
    with pytest.raises(ValueError):
        WorkflowManager(_placements([6.0, value], [50.0, 50.1]))

    with pytest.raises(ValueError):
        WorkflowManager(_placements([6.0, 6.1], [50.0, value]))


def test_init_error_names_the_offending_row():
    placements = _placements([6.0, 6.1, 999.0], [50.0, 50.1, 50.2])

    with pytest.raises(ValueError, match="2: 999.0"):
        WorkflowManager(placements)


def test_init_accepts_a_single_placement_at_the_origin():
    # the artificial extent width was multiplicative, so zero stayed zero
    man = WorkflowManager(_placements([0.0], [0.0]))

    assert man.ext.xMin < man.ext.xMax
    assert man.ext.yMin < man.ext.yMax


@pytest.mark.parametrize("lon,lat", [(0.0, 0.0), (-7.5, -33.2), (180.0, 90.0), (-180.0, -90.0)])
def test_init_accepts_the_coordinate_limits_and_orders_the_extent(lon, lat):
    man = WorkflowManager(_placements([lon], [lat]))

    assert man.ext.xMin < man.ext.xMax
    assert man.ext.yMin < man.ext.yMax


def test_init_accepts_a_valid_batch():
    man = WorkflowManager(_placements([-180.0, 0.0, 180.0], [-90.0, 0.0, 90.0]))

    assert man.placements.shape[0] == 3




def test_to_xarray_keeps_the_location_id_column():
    man = _loaded_manager()
    man.placements["location_id"] = [7, 9]

    first = man.to_xarray()

    assert "location_id" in man.placements.columns
    assert list(first["location"].values) == [7, 9]

    # the export must be repeatable
    second = man.to_xarray()
    assert list(second["location"].values) == [7, 9]


def test_to_xarray_does_not_change_the_given_output_variables():
    man = _loaded_manager()
    output_variables = ["lon", "capacity_factor"]

    man.to_xarray(output_variables=output_variables)

    assert output_variables == ["lon", "capacity_factor"]




def test_to_xarray_keeps_the_first_day_of_a_cross_year_simulation():
    man = _loaded_manager(periods=96, start="2019-12-30 00:00")
    days = np.unique(man.time_index.date)
    man.sim_data_daily = {"daily_output": np.arange(len(days) * 2, dtype=float).reshape(len(days), 2)}

    xds = man.to_xarray()

    assert xds["time_days"].shape == (len(days),)
    assert str(xds["time_days"].values[0])[:10] == "2019-12-30"
    assert np.allclose(xds["daily_output"].values, man.sim_data_daily["daily_output"])


def test_to_xarray_keeps_all_days_within_one_year():
    man = _loaded_manager(periods=72, start="2020-03-01 00:00")
    days = np.unique(man.time_index.date)
    man.sim_data_daily = {"daily_output": np.zeros((len(days), 2))}

    xds = man.to_xarray()

    assert xds["time_days"].shape == (len(days),)




def _write_result(path, location_ids):
    xds = xarray.Dataset(
        data_vars={"capacity_factor": (("time", "location"), np.full((3, len(location_ids)), 0.5))},
        coords={
            "time": pd.date_range("2020-01-01", periods=3, freq="h"),
            "location": list(location_ids),
        },
    )
    xds.to_netcdf(path)
    return path


def test_load_workflow_result_with_one_file(tmp_path):
    _write_result(tmp_path / "simulation_group_00000.nc", [2, 0, 1])

    ds = load_workflow_result(str(tmp_path))

    assert list(ds["location"].values) == [0, 1, 2]
    assert ds["capacity_factor"].shape == (3, 3)


def test_load_workflow_result_with_several_files(tmp_path):
    _write_result(tmp_path / "simulation_group_00000.nc", [3, 2])
    _write_result(tmp_path / "simulation_group_00001.nc", [1, 0])

    ds = load_workflow_result(str(tmp_path))

    assert list(ds["location"].values) == [0, 1, 2, 3]


def test_load_workflow_result_without_sorting(tmp_path):
    _write_result(tmp_path / "simulation_group_00000.nc", [2, 0, 1])

    ds = load_workflow_result(str(tmp_path), sortby=None)

    assert list(ds["location"].values) == [2, 0, 1]


def test_load_workflow_result_without_any_file(tmp_path):
    with pytest.raises(ValueError):
        load_workflow_result(str(tmp_path))



def tiny_workflow(placements, factor=1.0):
    """A minimal workflow which needs no weather source, for the distribution test."""
    man = WorkflowManager(placements)
    man.set_time_index(pd.date_range("2020-01-01", periods=3, freq="h"))
    man.sim_data["capacity_factor"] = np.full((3, placements.shape[0]), factor)
    return man.to_xarray()


def test_distribute_workflow_does_not_change_the_given_placements():
    placements = _placements([6.0, 6.1, 6.2, 6.3], [50.0, 50.1, 50.2, 50.3])
    before = placements.copy()

    xds = distribute_workflow(
        workflow_function=tiny_workflow,
        placements=placements,
        jobs=2,
        factor=0.5,
    )

    assert xds["location"].shape == (4,)
    pd.testing.assert_frame_equal(placements, before)
    assert "location_id" not in placements.columns
    assert list(placements.index) == [0, 1, 2, 3]
