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
from pathlib import Path

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
output_dir = Path(__file__).parent / "tmp"

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
# of the tiles written above and open its global horizontal irradiance file
# for the year we downloaded, just to see what's in it.

# %%
import xarray as xr

year = 2000  # we only downloaded 2000 above

# pick any one tile directory for that year (there may be several, depending
# on how many tiles the boundary box spans)
tile_dir = sorted((output_dir / "tiles").glob(f"*/*/*/{year}"))[0]
print("Example tile directory:", tile_dir)

# each variable is stored in its own file, named
# "<...>.z<zoom>.x<x-tile>.y<y-tile>.y<year>.<parameter>.nc"
parameter_file = sorted(tile_dir.glob("*surface_solar_radiation_downwards*.nc"))[0]
print("Example parameter file:", parameter_file)

with xr.open_dataset(parameter_file) as ds:
    print(ds)

# %% [markdown]
# ## Step 4 – Use the data in a real solar simulation workflow
#
# `execute_workflow_iteratively()` takes care of picking the right weather
# tile for each placement, so the tiled `era5_path` template from Step 1 can
# be handed straight to a RESKit solar workflow (here: `openfield_pv_era5`).
# Note that the tile template only resolves `<ZOOM>`/`<X-TILE>`/`<Y-TILE>` —
# the `<year>` subfolder (see Step 3) still needs to be appended manually,
# since each simulation run covers a single year of weather data. It also
# needs Global Solar Atlas (GSA) rasters to bias-correct the ERA5 irradiance;
# the small bundled `rk.TEST_DATA` rasters are used below as stand-ins — swap
# them for real GSA files (https://globalsolaratlas.info/) for an actual study.
#
# See `examples/4_solar/4_1_solar_workflows_overview.ipynb` for more on the
# workflow itself.

# %%
import pandas as pd

# Example: three locations in Germany, each with PV module specifications
placements = pd.DataFrame(
    {
        "lon": [6.004750, 6.1, 6.2],
        "lat": [50.784432, 50.8, 50.9],
        "capacity": [2500, 2500, 2500],
        "tilt": [38, 38, 38],
        "azimuth": [180, 180, 180],
        "elev": [300, 300, 300],
    }
)

reskit_xr = rk.execute_workflow_iteratively(
    workflow=rk.solar.openfield_pv_era5,
    weather_path_varname="era5_path",
    zoom=4,
    placements=placements,
    era5_path=str(Path(result["era5_path"]) / str(year)),
    global_solar_atlas_ghi_path=rk.TEST_DATA["gsa-ghi-like.tif"],
    global_solar_atlas_dni_path=rk.TEST_DATA["gsa-dni-like.tif"],
)
print(reskit_xr["capacity_factor"])
