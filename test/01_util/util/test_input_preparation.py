import ast
import inspect
import sys
import textwrap

import pytest

from reskit.weather import Era5Source
from reskit.weather.Era5Source.Era5Prepare import _ERA5_NC_TO_TILE_LABEL, era5_variables
from reskit.util.input_preparation import (
    _SOURCE_PREPARERS,
    _merge_dependencies,
    _prepare_gwa4,
    depends_on,
    download_and_process,
)


def _raw_variables_for_workflow(workflow):
    """Test helper: resolve a registered workflow's raw ERA5 tile variables via the
    Era5Source-owned mapping (mirrors what download_and_process does internally).
    """
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
    # same ERA5 needs as the CSP_PTR_ERA5 wrapper ...
    assert depends_on["CSP_PTR_ERA5_specific_dataset"]["ERA5"] == depends_on["CSP_PTR_ERA5"]["ERA5"]
    # ... but only the DNI raster (HTF selection, which needs TEMP, happens in the wrapper)
    assert depends_on["CSP_PTR_ERA5_specific_dataset"]["GSA"] == ["DNI"]


def test_merge_dependencies_single_workflow():
    # a single-element list reproduces that workflow's own dependencies
    merged = _merge_dependencies(["openfield_pv_era5"])
    assert merged == depends_on["openfield_pv_era5"]


def test_merge_dependencies_unions_across_workflows():
    merged = _merge_dependencies(["openfield_pv_era5", "CSP_PTR_ERA5", "wind_era5_PenaSanchezDunkelWinklerEtAl2025"])
    # union spans every source touched by the given workflows
    assert set(merged) == {"ERA5", "GSA", "GWA4"}
    # every variable of each input workflow is present in the union
    for workflow in ("openfield_pv_era5", "CSP_PTR_ERA5", "wind_era5_PenaSanchezDunkelWinklerEtAl2025"):
        for source, variables in depends_on[workflow].items():
            assert set(variables) <= set(merged[source])


def test_merge_dependencies_deduplicates_preserving_order():
    # openfield_pv and CSP share several ERA5 variables; the union must not repeat them
    merged = _merge_dependencies(["openfield_pv_era5", "CSP_PTR_ERA5"])
    for variables in merged.values():
        assert len(variables) == len(set(variables))


def test_merge_dependencies_unknown_workflow_raises_value_error():
    with pytest.raises(ValueError):
        _merge_dependencies(["not_a_real_workflow_name"])


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


_SNOW_CDS_NAMES = ["snow_albedo", "snow_density", "snow_depth", "snowfall"]


def test_the_snow_variables_stay_downloadable_and_tileable():
    """No workflow requires the snow data, but a user must be able to prepare it.

    openfield_pv_era5 read the four snow variables without using them, see BUG-18. The
    read is removed. Era5Source keeps its snow loaders, therefore the download and the
    tiling must keep their snow entries.
    """
    for cds_name in _SNOW_CDS_NAMES:
        assert cds_name in era5_variables
        assert cds_name in Era5Source.CDS_TO_NC_NAME
        assert Era5Source.CDS_TO_NC_NAME[cds_name] in _ERA5_NC_TO_TILE_LABEL

    assert set(Era5Source.raw_passthrough_variables(_SNOW_CDS_NAMES)) == {"asn", "rsn", "sd", "sf"}


def test_no_workflow_depends_on_the_snow_variables():
    """The snow data is not used by any calculation, so no workflow may download it."""
    for workflow, sources in depends_on.items():
        for cds_name in _SNOW_CDS_NAMES:
            assert cds_name not in sources.get("ERA5", []), f"{workflow} downloads unused snow data"


# What the ERA5 preprocessing derives from the raw download, see _ERA5_NC_TO_TILE_LABEL
# and preprocess_era5_data() in reskit/weather/Era5Source/Era5Prepare.py.
_DERIVED_NC_NAMES = {
    ("u100", "v100"): ["ws100", "wd100"],
    ("u10", "v10"): ["ws10", "wd10"],
    ("ssrd",): ["ssrd", "ssrd_t_adj"],
    ("fdir",): ["fdir", "fdir_t_adj"],
}

_ERA5_WORKFLOWS = sorted(workflow for workflow in depends_on if "ERA5" in depends_on[workflow])


def _prepared_nc_names(workflow):
    """Give the NC names which a prepared ERA5 dataset of the workflow contains."""
    cds_names = depends_on[workflow]["ERA5"]
    downloaded = {Era5Source.CDS_TO_NC_NAME[name] for name in cds_names if name in Era5Source.CDS_TO_NC_NAME}

    names = set(Era5Source.raw_passthrough_variables(cds_names))
    for inputs, outputs in _DERIVED_NC_NAMES.items():
        if set(inputs) <= downloaded:
            names.update(outputs)
    return names


def _workflow_function(workflow):
    """Find the function of a registered workflow name."""
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("reskit"):
            continue
        function = getattr(module, workflow, None)
        if callable(function) and getattr(function, "__module__", "").startswith("reskit"):
            return function
    raise AssertionError(f"no function found for the registered workflow {workflow!r}")


def _era5_variables_read_by(function):
    """Give the variables which the workflow reads from an ERA5 source."""
    variables = []
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function)))):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "read":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        source_type = keywords.get("source_type")
        if not isinstance(source_type, ast.Constant) or source_type.value != "ERA5":
            continue
        given = keywords.get("variables")
        if isinstance(given, ast.List):
            variables += [element.value for element in given.elts if isinstance(element, ast.Constant)]
    return variables


def _nc_name_of(variable):
    """Give the NC name which the standard loader of the variable reads.

    Give None if the loader builds the name itself, as the wind speed loaders do. Those
    loaders already raise an explicit RuntimeError when a component is absent.
    """
    loader = getattr(Era5Source, f"sload_{variable}", None)
    if loader is None:
        return None
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(loader)))):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "load":
            continue
        given = list(node.args) + [keyword.value for keyword in node.keywords if keyword.arg == "variable"]
        for argument in given:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                return argument.value
    return None


@pytest.mark.parametrize("workflow", _ERA5_WORKFLOWS)
def test_era5_dependencies_cover_the_variables_the_workflow_reads(workflow):
    """A prepared ERA5 dataset must contain every variable which the workflow reads (BUG-18)."""
    available = _prepared_nc_names(workflow)

    for variable in _era5_variables_read_by(_workflow_function(workflow)):
        nc_name = _nc_name_of(variable)
        if nc_name is None:
            continue
        assert nc_name in available, (
            f"{workflow} reads '{variable}' ('{nc_name}'), but depends_on['{workflow}']['ERA5'] "
            f"prepares only {sorted(available)}"
        )


@pytest.mark.parametrize("workflow", _ERA5_WORKFLOWS)
def test_every_era5_dependency_has_a_tile_label(workflow):
    """Every raw ERA5 variable of a workflow must have a tile label (BUG-18)."""
    for nc_name in Era5Source.raw_passthrough_variables(depends_on[workflow]["ERA5"]):
        assert nc_name in _ERA5_NC_TO_TILE_LABEL
