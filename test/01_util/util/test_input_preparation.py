from reskit.weather.Era5Source.Era5Prepare import _ERA5_NC_TO_TILE_LABEL
from reskit.util.input_preparation import _raw_variables_for_workflow, depends_on


def test_raw_variables_for_workflow_wind():
    result = _raw_variables_for_workflow("wind_era5_PenaSanchezDunkelWinklerEtAl2025")
    assert set(result) == {"t2m", "sp", "blh"}


def test_raw_variables_excludes_preprocessed():
    result = _raw_variables_for_workflow("wind_era5_PenaSanchezDunkelWinklerEtAl2025")
    # u100/v100 are consumed by preprocessing (→ ws100) and must not be re-tiled raw
    assert "u100" not in result
    assert "v100" not in result


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
