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
# - CDS account and `~/.cdsapirc` API key configured:
#   https://cds.climate.copernicus.eu/how-to-api
# - CDO installed: `conda install -c conda-forge cdo python-cdo`

# %%
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
# ├── raw/       ← CDS download + intermediate processed NC files
# └── tiles/     ← tiled output: <zoom>/<x-tile>/<y-tile>/<year>/*.nc
# ```
#
# Set `tiling=False` to skip step 3 and only download + preprocess.

# %%
output_dir = "/path/to/your/era5_data"  # <-- adjust this

result = rk.download_and_process(
    workflow="wind_era5_PenaSanchezDunkelWinklerEtAl2025",
    start_date="2000-01-01",
    end_date="2000-12-31",
    boundary_box={"north": 55, "south": 47, "west": 6, "east": 15},  # Germany
    output_dir=output_dir,
    tiling=True,
    zoom_level=4,  # 16×16 global tile grid, standard for RESKit ERA5 data
)

# %% [markdown]
# ## Step 2 – Inspect the result
#
# When `tiling=True`, `era5_path` is a path template with `<ZOOM>`, `<X-TILE>`,
# and `<Y-TILE>` placeholders. Pass it to `get_dataframe_with_weather_tilepaths()`
# to resolve tile paths for a set of locations.

# %%
print("era5_path template:", result["era5_path"])
# → /path/to/your/era5_data/tiles/<ZOOM>/<X-TILE>/<Y-TILE>

# %% [markdown]
# ## Step 3 – Load a tile with Era5Source
#
# Append the year to the template to point at a specific tile directory,
# then load it with `Era5Source`.

# %%
from reskit.util.weather_tile import get_dataframe_with_weather_tilepaths
import pandas as pd

# Example: a single location in Germany
placements = pd.DataFrame({"lat": [50.8], "lon": [6.1]})  # Aachen

placements = get_dataframe_with_weather_tilepaths(
    placements=placements,
    weather_path=result["era5_path"],
    zoom=4,
)
print(placements[["lat", "lon", "source"]])

# Load the tile for 2000 and inspect wind speed
from reskit.weather import Era5Source

tile_path = placements["source"].iloc[0] + "/2000"
src = Era5Source(source=tile_path)
src.sload_elevated_wind_speed()
print(src.data["elevated_wind_speed"])
