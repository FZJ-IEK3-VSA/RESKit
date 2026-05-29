import numpy as np
import pandas as pd

from reskit.hydro.workflows.hydro_workflow_manager import HydroWorkflowManager
from reskit.hydro.workflows import run_of_river_workflow, run_of_river_daily_discharge_workflow


def _placements():
    return pd.DataFrame(
        {
            "lon": [6.1, 6.2],
            "lat": [50.5, 50.6],
            "capacity": [1_000.0, 2_000.0],
        }
    )


def test_simulate_run_of_river_shapes_and_limits():
    inflow = np.array(
        [
            [10.0, 20.0],
            [5.0, 8.0],
        ]
    )

    wf = HydroWorkflowManager(_placements())
    out = wf.simulate_run_of_river(
        inflow_m3s=inflow,
        net_head_m=np.array([40.0, 60.0]),
        efficiency=0.9,
    ).sim_data

    assert out["power_output_w"].shape == inflow.shape
    assert out["capacity_factor"].shape == inflow.shape
    assert (out["capacity_factor"] >= 0).all()
    assert (out["capacity_factor"] <= 1).all()


def test_run_of_river_workflow_returns_dataset():
    placements = _placements()
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    inflow = np.array(
        [
            [10.0, 12.0],
            [11.0, 13.0],
            [9.0, 14.0],
        ]
    )

    ds = run_of_river_workflow(
        placements=placements,
        inflow_m3s=inflow,
        net_head_m=np.array([30.0, 35.0]),
        time_index=times,
        efficiency=0.9,
    )

    assert "capacity_factor" in ds
    assert "total_system_generation" in ds
    assert ds["capacity_factor"].shape == inflow.shape


def test_daily_discharge_workflow_can_uncap_production():
    placements = _placements()
    times = pd.date_range("2020-01-01", periods=2, freq="h")
    discharge = np.array(
        [
            [600000.0, 600000.0],
            [600000.0, 600000.0],
        ]
    )

    capped = run_of_river_daily_discharge_workflow(
        placements=placements,
        discharge_m3_per_day=discharge,
        net_head_m=np.array([50.0, 50.0]),
        time_index=times,
        efficiency=0.9,
        cap_production_by_capacity=True,
    )
    uncapped = run_of_river_daily_discharge_workflow(
        placements=placements,
        discharge_m3_per_day=discharge,
        net_head_m=np.array([50.0, 50.0]),
        time_index=times,
        efficiency=0.9,
        cap_production_by_capacity=False,
    )

    assert np.all(uncapped["total_system_generation"].values >= capped["total_system_generation"].values)
    assert np.any(uncapped["capacity_factor"].values > 1.0)
