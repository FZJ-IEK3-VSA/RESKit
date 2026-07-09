# %% [markdown]
# # Prepare ERA5 Data for Wind Workflow
#
# This example shows how to use RESKit's reproducible ERA5 preparation workflow.
# It downloads ERA5 data from the Copernicus Climate Data Store (CDS), applies
# preprocessing (wind speed from u/v components, solar unit conversions), and
# optionally tiles the result into the `<zoom>/<x>/<y>/<year>/` directory structure
# expected by `Era5Source` and `execute_workflow_iteratively()`.
#
# **Prerequisites**
# - CDS account and `~/.cdsapirc` API key configured (for instructions see link below):
#   https://cds.climate.copernicus.eu/how-to-api
#   (processing is done with xarray/netCDF4 — no external CDO binary required)

# %%
from pathlib import Path

import reskit as rk

# %% [markdown]
# ## Step 1 – Download, process, and tile in one call
#
# `download_and_process` orchestrates all three steps:
# 1. Downloads the ERA5 variables required by the chosen workflow from CDS
# 2. Preprocesses them (ws100 from u/v components, solar unit + time-shift corrections)
# 3. Splits the processed files into the tiled directory structure
#
# Output layout under `output_dir`:
# ```
# output_dir/
# ├── raw/        ← CDS download files only
# ├── processed/  ← usable processed NC files (ws100, solar_t_adj, …)
# └── tiles/      ← tiled output: <zoom>/<x-tile>/<y-tile>/<year>/*.nc
# ```
#
# Set `tiling=False` to skip step 3; `era5_path` will then point to `processed/`
# and can be used directly with Era5Source.
#
# The wind workflow also depends on Global Wind Atlas (GWA) data. That source cannot
# be downloaded automatically yet, so `download_and_process` prepares the ERA5 data
# and just prints a note reminding you to fetch the GWA rasters manually (see Step 4).

# %%
output_dir = Path(__file__).parent / "tmp"

result = rk.download_and_process(
    workflow="wind_era5_PenaSanchezDunkelWinklerEtAl2025",
    start_date="2000-01-01",
    end_date="2000-01-03",
    boundary_box={"north": 55, "south": 47, "west": 6, "east": 15},  # Germany
    output_dir=output_dir,
    tiling=True,
    zoom_level=4,  # 16×16 global tile grid, standard for RESKit ERA5 data
)

# you can check the download status of the raw data at: 
# https://cds.climate.copernicus.eu/requests

# NOTE: If you intend to use this outside of example, please make sure to always download a full year.

# %% [markdown]
# ## Step 2 – Inspect the result
#
# When `tiling=True`, `era5_path` is a path template with `<ZOOM>`, `<X-TILE>`,
# and `<Y-TILE>` placeholders. Pass it to `get_dataframe_with_weather_tilepaths()`
# to resolve tile paths for a set of locations.

# %%
print("era5_path template:", result["era5_path"])
# → <script_dir>/tmp/tiles/<ZOOM>/<X-TILE>/<Y-TILE>

# %% [markdown]
# ## Step 3 – Browse the downloaded files directly
#
# For later simulation, you can use either the `processed/` or the tiled
# `tiles/` output (if chosen in Step 1) — see the `output_dir`
# layout described above. The `raw/` files are kept only for reference or
# further processing and should not be fed into the simulation workflows.
#
# With tiling enabled, the data is written as one `.nc` file per variable
# per year, under `tiles/<zoom>/<x-tile>/<y-tile>/<year>/`. Below we pick one
# of the tiles written above and open its wind speed file for the year we
# downloaded, just to see what's in it.

# %%
import xarray as xr

year = 2000  # we only downloaded 2000 above

# pick any one tile directory for that year (there may be several, depending
# on how many tiles the boundary box spans)
tile_dir = sorted((output_dir / "tiles").glob(f"*/*/*/{year}"))[0]
print("Example tile directory:", tile_dir)

# each variable is stored in its own file, named
# "<...>.z<zoom>.x<x-tile>.y<y-tile>.y<year>.<parameter>.nc"
parameter_file = sorted(tile_dir.glob("*100m_wind_speed*.nc"))[0]
print("Example parameter file:", parameter_file)

with xr.open_dataset(parameter_file) as ds:
    print(ds)

# %% [markdown]
# ## Step 4 – Use the data in a real wind simulation workflow
#
# `execute_workflow_iteratively()` takes care of picking the right weather
# tile for each placement, so the tiled `era5_path` template from Step 1 can
# be handed straight to a RESKit wind workflow (here:
# `wind_era5_PenaSanchezDunkelWinklerEtAl2025`). Note that the tile template
# only resolves `<ZOOM>`/`<X-TILE>`/`<Y-TILE>` — the `<year>` subfolder
# (see Step 3) still needs to be appended manually, since each simulation
# run covers a single year of weather data. It also needs Global Wind Atlas
# (GWA) rasters to bias-correct the ERA5 wind speeds; the small bundled
# `rk.TEST_DATA` rasters are used below as stand-ins — swap them for real
# GWA files (https://globalwindatlas.info/) for an actual study.
#
# See `examples/3_wind/3_7_example_ethos_reskit_wind_workflow.ipynb` for
# more on the workflow itself.

# %%
import pandas as pd

# Example: three locations in Germany, each with turbine specifications
placements = pd.DataFrame(
    {
        "lon": [6.004750, 6.1, 6.2],
        "lat": [50.784432, 50.8, 50.9],
        "hub_height": [120, 120, 82],
        "capacity": [4000, 4000, 4000],
        "rotor_diam": [130, 150, 136],
    }
)

reskit_xr = rk.execute_workflow_iteratively(
    workflow=rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025,
    weather_path_varname="era5_path",
    zoom=4,
    placements=placements,
    era5_path=str(Path(result["era5_path"]) / str(year)),
    gwa_100m_path=rk.TEST_DATA["gwa100-like.tif"],
    height_scaling_data={
        50: rk.TEST_DATA["gwa50-like.tif"],
        200: rk.TEST_DATA["gwa200-like.tif"],
    },
)
print(reskit_xr["capacity_factor"])
