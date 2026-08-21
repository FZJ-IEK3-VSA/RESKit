import numpy as np
import pandas as pd
import pytest

from reskit.hydro.workflows.hydro_workflow_manager import HydroWorkflowManager


def _placements():
    return pd.DataFrame({"lon": [6.1], "lat": [50.5]})


def _parflow_options():
    return {
        "root_dir": "unused",
        "alluvium_mask_file": "unused",
        "indicator_file": "unused",
    }


def _fake_extraction(*args, **kwargs):
    daily_volume = np.array([[0.0, 86400.0, 172800.0]])
    return {
        "selected_discharge_m3_per_day": daily_volume,
        "selected_candidate_idx": np.array([0]),
        "selected_from_alluvium": np.array([True]),
        "selected_cell_overview": [{}],
    }


def test_extract_discharge_converts_units_and_interpolates(monkeypatch):
    monkeypatch.setattr(
        "reskit.hydro.workflows.hydro_workflow_manager.extract_selected_discharge_alluvium",
        _fake_extraction,
    )
    times = pd.date_range("2020-01-01", "2020-01-02", freq="12h")
    wf = HydroWorkflowManager(_placements())

    with pytest.warns(UserWarning, match="linearly interpolated"):
        wf.extract_discharge("parflow", times, _parflow_options())

    assert np.allclose(wf.sim_data["discharge_m3s"][:, 0], [0.0, 0.5, 1.0])
    assert wf.workflow_parameters["temporal_resampling_applied"] == "True"


def test_extract_discharge_rejects_unknown_product():
    times = pd.date_range("2020-01-01", periods=2, freq="D")
    wf = HydroWorkflowManager(_placements())

    with pytest.raises(ValueError, match="Unknown discharge product"):
        wf.extract_discharge("unknown", times)


def test_extract_native_three_hour_parflow_product(monkeypatch):
    captured = {}

    def fake_extraction(*args, **kwargs):
        captured.update(kwargs)
        volume = np.array([[0.0, 10800.0, 21600.0]])
        return {
            "selected_discharge_m3_per_timestep": volume,
            "selected_candidate_idx": np.array([0]),
            "selected_from_alluvium": np.array([True]),
            "selected_cell_overview": [{}],
        }

    monkeypatch.setattr(
        "reskit.hydro.workflows.hydro_workflow_manager.extract_selected_discharge_alluvium",
        fake_extraction,
    )
    times = pd.date_range("2020-01-01", periods=3, freq="3h")
    wf = HydroWorkflowManager(_placements())

    wf.extract_discharge("parflow-3hour", times, _parflow_options())

    assert np.allclose(wf.sim_data["discharge_m3s"][:, 0], [0.0, 1.0, 2.0])
    assert "tmp_ice2" in captured["data_url"]
    assert "3hours_20200101-20201231.ICE2.nc" in captured["data_url"]
    assert wf.workflow_parameters["temporal_resampling_applied"] == "False"


def test_parflow_alias_uses_daily_product(monkeypatch):
    captured = {}

    def fake_extraction(*args, **kwargs):
        captured.update(kwargs)
        return _fake_extraction()

    monkeypatch.setattr(
        "reskit.hydro.workflows.hydro_workflow_manager.extract_selected_discharge_alluvium",
        fake_extraction,
    )
    times = pd.date_range("2020-01-01", periods=2, freq="D")
    wf = HydroWorkflowManager(_placements())

    wf.extract_discharge("parflow", times, _parflow_options())

    assert "ParFlow-DE06-HC_v03" in captured["data_url"]
    assert "1day_20200101-20201231.nc" in captured["data_url"]
    assert wf.workflow_parameters["discharge_product"] == "parflow-1day"
