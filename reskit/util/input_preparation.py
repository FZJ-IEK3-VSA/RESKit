import os
import reskit as rk

depends_on = {
    "wind_era5_PenaSanchezDunkelWinklerEtAl2025": {
        "GWA4": ["50m", "100m", "200m"],
        "ERA5": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "2m_temperature",
            "surface_pressure",
            "boundary_layer_height",
        ],
    },
    "kacke": {
        "ERA5": [
            "2m_temperature",
        ],
    },
}

# Maps CDS API variable names to their short names inside the downloaded NetCDF file.
_era5_cds_to_nc_name = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "2m_dewpoint_temperature": "d2m",
    "2m_temperature": "t2m",
    "surface_pressure": "sp",
    "boundary_layer_height": "blh",
    "forecast_surface_roughness": "fsr",
    "surface_solar_radiation_downwards": "ssrd",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
}

# Variables whose NC short names are consumed by preprocess_era5_data and replaced
# by derived outputs (e.g. u100+v100 → ws100). These should not be re-tiled as raw.
_era5_preprocessed_nc_names = {"u10", "v10", "u100", "v100", "ssrd", "fdir"}


def _raw_variables_for_workflow(workflow: str) -> list:
    """Return the NC short names that need to be tiled from the raw download file
    for a given workflow — i.e. the variables that are not replaced by preprocessing."""
    era5_cds_names = depends_on[workflow].get("ERA5", [])
    return [
        _era5_cds_to_nc_name[cds_name]
        for cds_name in era5_cds_names
        if cds_name in _era5_cds_to_nc_name and _era5_cds_to_nc_name[cds_name] not in _era5_preprocessed_nc_names
    ]


def download_and_process(
    workflow,
    start_date,
    end_date,
    boundary_box,
    output_dir,
    tiling=False,
    zoom_level=4,
    tile_output_dir=None,
):
    if workflow == "wind_era5_PenaSanchezDunkelWinklerEtAl2025":
        _tile_out = tile_output_dir or os.path.join(output_dir, "tiles")
        era5_path = rk.prepare_era5(
            start_date=start_date,
            end_date=end_date,
            boundary_box=boundary_box,
            output_dir=output_dir,
            variables=depends_on["wind_era5_PenaSanchezDunkelWinklerEtAl2025"]["ERA5"],
            tiling=tiling,
            zoom_level=zoom_level,
            tile_output_dir=_tile_out,
            raw_variables=_raw_variables_for_workflow(workflow),
        )
        if tiling:
            # return a path template for get_dataframe_with_weather_tilepaths()
            era5_path = os.path.join(era5_path, "<ZOOM>", "<X-TILE>", "<Y-TILE>")

        print(
            f"ERA5 data prepared and tiled at: {era5_path}. Please download data for height scaling from the Global Wind Atlas: https://globalwindatlas.info/en/download/gis-files"
        )

        return {
            "era5_path": era5_path,
            "height_scaling_data": {
                50: "Please download directly from https://globalwindatlas.info/en/download/gis-files",
                100: "Please download directly from https://globalwindatlas.info/en/download/gis-files",
                200: "Please download directly from https://globalwindatlas.info/en/download/gis-files",
            },
        }
    elif workflow == "kacke":
        _tile_out = tile_output_dir or os.path.join(output_dir, "tiles")
        era5_path = rk.prepare_era5(
            start_date=start_date,
            end_date=end_date,
            boundary_box=boundary_box,
            output_dir=output_dir,
            variables=depends_on["kacke"]["ERA5"],
            tiling=tiling,
            zoom_level=zoom_level,
            tile_output_dir=_tile_out,
            raw_variables=_raw_variables_for_workflow(workflow),
        )
        if tiling:
            # return a path template for get_dataframe_with_weather_tilepaths()
            era5_path = os.path.join(era5_path, "<ZOOM>", "<X-TILE>", "<Y-TILE>")


        return {
            "era5_path": era5_path,
        }

    raise ValueError(f"Unknown workflow: {workflow}")
