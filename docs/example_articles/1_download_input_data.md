# Obtain and prepare RESKit input data

For catalogued workflow inputs, start with
[Get input data from the ETHOS.Data catalogue](../how_to/get_input_data.md).
It uses `reskit.data` and `reskit-data` to select, fetch and verify RESKit's inputs.

The examples below cover obtaining upstream weather data and preparing it for
RESKit. Use [development staging](../how_to/get_input_data.md#develop-against-unpublished-data)
when testing a prepared dataset before catalogue acceptance.

## Time-resolved weather data

### ERA5

[ERA5](https://doi.org/10.24381/cds.adbb2d47) (ECMWF Reanalysis v5) is RESKit's main
weather source for the wind, PV, CSP and DAC workflows. RESKit reads it on the native
0.25° grid in one of two formats:

| | Zarr store | NetCDF4 files |
|---|---|---|
| Source class | `Era5ZarrSource` | `Era5Source` |
| Selected when | path ends in `.zarr` or starts with `gs://` | otherwise |
| Download needed | no, read remotely chunk by chunk | yes, from the Copernicus CDS |
| Preprocessing | done on the fly | `download_and_process` / `prepare_era5` |

We recommend Zarr for remote and multi-user access: only the needed chunks are read.

#### Option A: read ERA5 from a Zarr store

1. Create an account for the
   [Earth Data Hub ERA5 single-level dataset](https://earthdatahub.destine.eu/collections/era5/datasets/reanalysis-era5-single-levels)
   and save the credentials in `~/.netrc`.
2. Pass the store URL as `era5_path` and limit the period with `time_slice`.
   Without it, a multi-year store is read in full.
3. See [Read ERA5 from a Zarr store](../examples/3_wind/3_8_use_workflows_with_zarr.ipynb).

Only stores on a regular lat/lon grid are supported.

#### Option B: download ERA5 as NetCDF4

1. Create a [CDS account](https://cds.climate.copernicus.eu/how-to-api), accept the
   ERA5 licence and put your API key in `~/.cdsapirc`.
2. Run `rk.download_and_process()` for your workflow, period and bounding box. Make the
   bounding box extend at least 1° beyond your outermost placements on every side.
   Otherwise the simulation stops with "Insufficient data". Outside of quick tests,
   download whole calendar years. The function downloads only the variables the
   workflow needs and preprocesses them:
   - wind speed `ws100`/`ws10` from the u/v components,
   - irradiance `ssrd_t_adj`/`fdir_t_adj`: J m⁻² → W m⁻², shifted +1 h so each value is
     the mean over the following hour. Solar workflows require this step.
3. Optionally set `tiling=True` to split the data into
   `tiles/<zoom>/<x>/<y>/<year>/` and run with `rk.execute_workflow_iteratively()`.
4. Pass `processed/` or the tile template as `era5_path`. Don't use `raw/`.

Examples:
- [Prepare ERA5 for wind workflows](../examples/1_load_input_data/1_1_3_prepare_era5_for_wind_workflow.ipynb)
- [Prepare ERA5 for solar workflows](../examples/1_load_input_data/1_1_4_prepare_era5_for_solar_workflow.ipynb)
- Background, manual route: [Download ERA5 with cdsapi](../examples/1_load_input_data/1_1_1_how_to_download_era5_data.ipynb),
  [Wind speed from u/v](../examples/1_load_input_data/1_1_2_wind_speed_from_vectors_in_era5.ipynb)

You don't need to download the long-run-average rasters (`Era5Source.LONG_RUN_AVERAGE_*`).
They are included with RESKit.

### MERRA-2

[MERRA-2](https://doi.org/10.1175/JCLI-D-11-00015.1) (Modern-Era Retrospective Analysis
for Research and Applications):

- [Show the structure of MERRA data](../examples/1_load_input_data/1_2_1_reading_merra_weather_data.ipynb)
- [Calculate wind speeds from MERRA data](../examples/1_load_input_data/1_2_2_vertically_project_wind_speed_from_merra.ipynb)
<!-- 3. ICONLAM # TODO Create Example 
1. COSOMO # Needs check for relevance # TODO Create Example -->

<!-- ## Globally spatially Resolved Irrdatioan and Windspeed Data

1. Global Solar Atlas (GSA) # No Example
2. Global Wind Atlas  (GWA) # No Example -->

# Wind Power Curves 
A wind turbine power curve is a chart that shows how much power a turbine generates at various wind speeds.

1. Power Curves from https://www.thewindpower.net/ can be integrated in RESKit as shown in this [example](../examples/1_load_input_data/1_3_1_process_power_curves_from_thewindpower_net.ipynb)
