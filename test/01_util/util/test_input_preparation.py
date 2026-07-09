import pytest

from reskit.weather import Era5Source
from reskit.weather.Era5Source.Era5Prepare import _ERA5_NC_TO_TILE_LABEL
from reskit.util.input_preparation import (
    ALL_ERA5_WORKFLOWS,
    _NON_WORKFLOW_KEYS,
    _SOURCE_PREPARERS,
    _known_reskit_workflows,
    _prepare_gwa4,
    depends_on,
    download_and_process,
)


def _raw_variables_for_workflow(workflow):
    """Test helper: resolve a registered workflow's raw ERA5 tile variables via the
    Era5Source-owned mapping (mirrors what download_and_process does internally)."""
    return Era5Source.raw_passthrough_variables(depends_on[workflow]["ERA5"])

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


def test_dac_workflows_registered():
    for workflow in ("ht_dac_era5_wenzel2025", "lt_dac_era5_wenzel2025"):
        assert workflow in depends_on
        # DAC only needs raw 2m air + dew temperature, no preprocessed variables
        assert set(_raw_variables_for_workflow(workflow)) == {"t2m", "d2m"}


def test_cooling_heating_workflows_registered():
    expected = {
        "air_cooling_wenzel2025": {"t2m"},
        "air_source_heat_pump": {"t2m"},
        "evaporative_cooling_wortmann2025": {"t2m", "d2m"},
    }
    for workflow, raw in expected.items():
        assert workflow in depends_on
        assert set(_raw_variables_for_workflow(workflow)) == raw


def test_csp_specific_dataset_registered():
    assert "CSP_PTR_ERA5_specific_dataset" in depends_on
    assert depends_on["CSP_PTR_ERA5_specific_dataset"] == depends_on["CSP_PTR_ERA5"]


def test_all_era5_workflows_is_union():
    union = set(depends_on[ALL_ERA5_WORKFLOWS]["ERA5"])
    # the meta-workflow must cover every variable required by every real workflow
    for workflow, deps in depends_on.items():
        if workflow in _NON_WORKFLOW_KEYS:
            continue
        assert set(deps.get("ERA5", [])) <= union, f"{workflow} not covered by {ALL_ERA5_WORKFLOWS}"


def test_all_era5_workflows_has_no_duplicates():
    era5 = depends_on[ALL_ERA5_WORKFLOWS]["ERA5"]
    assert len(era5) == len(set(era5))


def test_all_era5_workflows_raw_variables_resolve():
    # every union variable maps cleanly; raw passthroughs are exactly the non-preprocessed ones
    assert set(_raw_variables_for_workflow(ALL_ERA5_WORKFLOWS)) == {"t2m", "sp", "blh", "d2m"}


def test_unsupported_known_workflow_raises_not_implemented():
    # a real RESKit workflow that is not (yet) registered in depends_on
    unsupported = next(wf for wf in _known_reskit_workflows() if wf not in depends_on)
    with pytest.raises(NotImplementedError):
        download_and_process(unsupported, **_DUMMY_KWARGS)


def test_unknown_workflow_raises_value_error():
    with pytest.raises(ValueError):
        download_and_process("not_a_real_workflow_name", **_DUMMY_KWARGS)


def test_gwa4_preparer_notifies_and_returns_none(capsys):
    # GWA4 has no automated download: its preparer must not fail, must return None,
    # and must tell the user to download the rasters manually
    assert _prepare_gwa4(depends_on["wind_era5_PenaSanchezDunkelWinklerEtAl2025"]["GWA4"]) is None
    out = capsys.readouterr().out
    assert "globalwindatlas" in out.lower()


def test_gwa4_source_registered_as_callable():
    # GWA4 is an explicitly registered (placeholder) preparer, not a missing/None entry
    assert callable(_SOURCE_PREPARERS["GWA4"])


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
