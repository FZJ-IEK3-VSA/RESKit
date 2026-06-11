import pytest

from reskit.weather.Era5Source.Era5Prepare import _ERA5_NC_TO_TILE_LABEL
from reskit.util.input_preparation import (
    _known_reskit_workflows,
    _raw_variables_for_workflow,
    depends_on,
    download_and_process,
)

_DUMMY_KWARGS = dict(
    start_date="2000-01-01",
    end_date="2000-01-31",
    boundary_box={"north": 1, "south": 0, "west": 0, "east": 1},
    output_dir="/tmp/does_not_matter",
)


def test_raw_variables_for_workflow_wind():
    result = _raw_variables_for_workflow("wind_era5_PenaSanchezDunkelWinklerEtAl2025")
    assert set(result) == {"t2m", "sp", "blh"}


def test_raw_variables_excludes_preprocessed():
    result = _raw_variables_for_workflow("wind_era5_PenaSanchezDunkelWinklerEtAl2025")
    # u100/v100 are consumed by preprocessing (→ ws100) and must not be re-tiled raw
    assert "u100" not in result
    assert "v100" not in result


def test_raw_variables_for_workflow_pv():
    result = _raw_variables_for_workflow("openfield_pv_era5")
    # ssrd/fdir (→ time-adjusted solar) and u10/v10 (→ ws10) are preprocessed, not raw
    assert set(result) == {"sp", "t2m", "d2m"}


def test_raw_variables_for_workflow_csp():
    result = _raw_variables_for_workflow("CSP_PTR_ERA5")
    # CSP only passes 2m_temperature through raw; fdir and u10/v10 are preprocessed
    assert set(result) == {"t2m"}


def test_raw_variables_solar_excludes_preprocessed():
    for workflow in ("openfield_pv_era5", "CSP_PTR_ERA5"):
        result = _raw_variables_for_workflow(workflow)
        for preprocessed in ("ssrd", "fdir", "u10", "v10"):
            assert preprocessed not in result


def test_solar_workflows_registered():
    for workflow in ("openfield_pv_era5", "CSP_PTR_ERA5"):
        assert workflow in depends_on
        assert depends_on[workflow]["ERA5"]


def test_unsupported_known_workflow_raises_not_implemented():
    # a real RESKit workflow that is not (yet) registered in depends_on
    unsupported = next(wf for wf in _known_reskit_workflows() if wf not in depends_on)
    with pytest.raises(NotImplementedError):
        download_and_process(unsupported, **_DUMMY_KWARGS)


def test_unknown_workflow_raises_value_error():
    with pytest.raises(ValueError):
        download_and_process("not_a_real_workflow_name", **_DUMMY_KWARGS)


def test_raw_variables_matches_depends_on():
    # every returned NC name must trace back to a CDS name in depends_on
    era5_cds_names = depends_on["wind_era5_PenaSanchezDunkelWinklerEtAl2025"]["ERA5"]
    result = _raw_variables_for_workflow("wind_era5_PenaSanchezDunkelWinklerEtAl2025")
    assert len(result) <= len(era5_cds_names)


def test_ERA5_NC_TO_TILE_LABEL_has_processed_vars():
    for var in ["ws100", "ws10", "ssrd_t_adj", "fdir_t_adj"]:
        assert var in _ERA5_NC_TO_TILE_LABEL, f"Missing processed var: {var}"


def test_ERA5_NC_TO_TILE_LABEL_has_raw_vars():
    for var in ["t2m", "sp", "blh", "d2m", "fsr"]:
        assert var in _ERA5_NC_TO_TILE_LABEL, f"Missing raw var: {var}"


def test_ERA5_NC_TO_TILE_LABEL_ws100_label():
    assert _ERA5_NC_TO_TILE_LABEL["ws100"] == "100m_wind_speed.processed"


def test_ERA5_NC_TO_TILE_LABEL_ssrd_t_adj_label():
    assert _ERA5_NC_TO_TILE_LABEL["ssrd_t_adj"] == "surface_solar_radiation_downwards.processed.t_adjusted"
