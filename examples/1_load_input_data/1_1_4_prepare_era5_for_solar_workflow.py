# %% [markdown]
# # Prepare ERA5 Data for Solar Workflows
#
# This example shows how to use RESKit's reproducible ERA5 preparation workflow
# for the ERA5-based solar workflows (`openfield_pv_era5` and `CSP_PTR_ERA5`).
# It downloads ERA5 data from the Copernicus Climate Data Store (CDS), applies
# preprocessing (solar unit + time-shift corrections, surface wind speed from
# u/v components), and optionally tiles the result into the
# `<zoom>/<x>/<y>/<year>/` directory structure expected by `Era5Source` and
# `execute_workflow_iteratively()`.
#
# **Prerequisites**
# - CDS account and `~/.cdsapirc` API key configured:
#   https://cds.climate.copernicus.eu/how-to-api
#   (processing is done with xarray/netCDF4 — no external CDO binary required)

# %%
import reskit as rk

# %% [markdown]
# ## Step 1 – Download, process, and tile in one call
#
# `download_and_process` orchestrates all three steps:
# 1. Downloads the ERA5 variables required by the chosen workflow from CDS
# 2. Preprocesses them (time-adjusted GHI/DHI in W m⁻², surface wind speed)
# 3. Splits the processed files into the tiled directory structure
#
# For the PV workflow (`openfield_pv_era5`) this prepares time-adjusted
# global/direct horizontal irradiance, surface wind speed, surface pressure,
# and 2 m air/dew temperature. The CSP workflow (`CSP_PTR_ERA5`) needs a subset
# (direct horizontal irradiance, surface wind speed, 2 m air temperature).
#
# Output layout under `output_dir`:
# ```
# output_dir/
# ├── raw/        ← CDS download files only
# ├── processed/  ← usable processed NC files (solar_t_adjusted, ws10, …)
# └── tiles/      ← tiled output: <zoom>/<x-tile>/<y-tile>/<year>/*.nc
# ```
#
# Set `tiling=False` to skip step 3; `era5_path` will then point to `processed/`
# and can be used directly with Era5Source.

# %%
output_dir = "/path/to/your/era5_data"  # <-- adjust this

result = rk.download_and_process(
    workflow="openfield_pv_era5",  # or "CSP_PTR_ERA5"
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

# Load the tile for 2000 and inspect the solar irradiance variables
from reskit.weather import Era5Source

tile_path = placements["source"].iloc[0] + "/2000"
src = Era5Source(source=tile_path)
src.sload_global_horizontal_irradiance()
src.sload_direct_horizontal_irradiance()
print(src.data["global_horizontal_irradiance"])
print(src.data["direct_horizontal_irradiance"])
