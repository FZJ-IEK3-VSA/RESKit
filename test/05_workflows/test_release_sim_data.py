import numpy as np
import pandas as pd
import pytest

from reskit import WorkflowManager


def _manager(time_selection: np.ndarray | None = None) -> WorkflowManager:
    placements = pd.DataFrame({"lon": [6.1, 6.2], "lat": [50.5, 50.6]})
    manager = WorkflowManager(placements)

    time_index = pd.date_range("2020-01-01", periods=4, freq="h")
    manager.set_time_index(time_index)

    if time_selection is None:
        time_step_count = 4
    else:
        time_step_count = int(np.sum(time_selection))
    manager._time_sel_ = time_selection

    placement_count = 2
    values_a = np.arange(time_step_count * placement_count, dtype=float)
    manager.sim_data["a"] = values_a.reshape(time_step_count, placement_count)
    manager.sim_data["b"] = np.ones((time_step_count, placement_count), dtype=np.float32)
    return manager


def test_release_sim_data_keeps_needed_and_requested_variables() -> None:
    manager = _manager()
    manager.sim_data["c"] = np.zeros((4, 2))

    # without output_variables, everything is part of the output
    manager.release_sim_data([], None)
    assert list(manager.sim_data) == ["a", "b", "c"]

    manager.release_sim_data(["a"], output_variables="c")
    assert list(manager.sim_data) == ["a", "c"]


@pytest.mark.parametrize("time_selection", [None, np.array([False, True, True, False])])
@pytest.mark.parametrize("output_variables", [None, ["a"]])
def test_to_xarray_release_gives_the_same_result(
    time_selection: np.ndarray | None,
    output_variables: list[str] | None,
) -> None:
    copying_manager = _manager(time_selection)
    expected = copying_manager.to_xarray(output_variables=output_variables)

    releasing_manager = _manager(time_selection)
    result = releasing_manager.to_xarray(output_variables=output_variables, release=True)

    assert result.identical(expected)
    assert len(releasing_manager.sim_data) == 0
