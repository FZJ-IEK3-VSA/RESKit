"""Regression tests for the parameter csv schema validation (BUG-05)."""

import pandas as pd
import pytest

from reskit import TEST_DATA
from reskit.parameters.parameters import OnshoreParameters


def _dummy_frame():
    return pd.read_csv(TEST_DATA["baseline_turbine_testdummy.csv"])


def test_unknown_column_raises(tmp_path):
    frame = _dummy_frame()
    frame["not_a_parameter"] = 1.0
    fp = tmp_path / "params.csv"
    frame.to_csv(fp, index=False)

    with pytest.raises(AttributeError, match="not_a_parameter"):
        OnshoreParameters(fp=str(fp), year=2030)


def test_unit_column_of_an_unknown_parameter_raises(tmp_path):
    frame = _dummy_frame()
    frame["gearbox_unit"] = "kg"
    fp = tmp_path / "params.csv"
    frame.to_csv(fp, index=False)

    with pytest.raises(AttributeError, match="gearbox_unit"):
        OnshoreParameters(fp=str(fp), year=2030)


def test_unit_columns_of_known_parameters_are_accepted(tmp_path):
    # 'tcc_share_unit' must resolve to the parameter 'tcc_share'. str.strip('_unit')
    # removed the leading 't' as well, so the suffix must be matched exactly.
    frame = _dummy_frame()
    assert "tcc_share_unit" in frame.columns
    fp = tmp_path / "params.csv"
    frame.to_csv(fp, index=False)

    params = OnshoreParameters(fp=str(fp), year=2030)

    assert params.base_rotor_diam == 118


def test_nameless_index_column_is_ignored(tmp_path):
    # pandas writes the row index as a nameless column, which is not a parameter
    frame = _dummy_frame()
    fp = tmp_path / "params.csv"
    frame.to_csv(fp, index=True)

    params = OnshoreParameters(fp=str(fp), year=2030)

    assert params.base_rotor_diam == 118


def test_missing_mandatory_column_raises(tmp_path):
    frame = _dummy_frame().drop(columns=["base_rotor_diam"])
    fp = tmp_path / "params.csv"
    frame.to_csv(fp, index=False)

    with pytest.raises(AttributeError, match="base_rotor_diam"):
        OnshoreParameters(fp=str(fp), year=2030)


def test_remarks_column_is_accepted(tmp_path):
    frame = _dummy_frame()
    assert "remarks" in frame.columns
    fp = tmp_path / "params.csv"
    frame.to_csv(fp, index=False)

    assert OnshoreParameters(fp=str(fp), year=2030).base_rotor_diam == 118
