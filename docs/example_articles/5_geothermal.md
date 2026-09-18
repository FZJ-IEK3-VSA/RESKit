# Geothermal

RESKit provides an Enhanced Geothermal System workflow in the
[geothermal example](../examples/5_geothermal/5_run_geothermal_simulations.ipynb).
The method is described by Franzmann, Heinrichs and Stolten (2025),
*Global geothermal electricity potentials: A technical, economic, and thermal
renewability assessment*, Renewable Energy 250, 123199,
[doi:10.1016/j.renene.2025.123199](https://doi.org/10.1016/j.renene.2025.123199).

The resource temperature model is equation (1) in section 2.3. It combines
Goutorbe's similarity-method heat flow and province-based heat production,
GLiM rock classes converted to conductivity, and NASA POWER surface temperature.
The archived sustainable-flow grid uses `q - 9000*A`; this particular correction
comes from Franzmann's original script and is not specified in the paper's
section 2.4.2, equation (5).

The `reskit.geothermal.preprocessing.create_geothermal_resource` function
reconstructs both resource files from the original inputs. It returns an xarray
Dataset and optionally writes the two NetCDF files accepted by `egs_workflow`.
Every new output includes units, source-file SHA-256 checksums, the paper DOI
and notes distinguishing the paper's method from the archived implementation.

The ICE-2 catalogue registers the following inputs, also selected by RESKit's
`geothermal_preprocessing` data collection:

| Dataset | Required file |
| --- | --- |
| `goutorbe-global-heat-flow-2011` | `Supplementary material.txt` |
| `global-lithological-map-glim-v1` | `glim_wgs84_0point5deg.txt.asc` |
| `nasa-power-geothermal-climatology` | `POWER_Global_Climatology_LST_2.nc` |
| `geothermal-conductivity-assumptions` | `Mapper_Lithologic_to_heat_conductivity.xlsx` |

These entries are registered in the internal source catalogue. Use the original
files already available on shared storage; remote fetching requires the entries
and their bytes to be distributed through a catalogue available to the caller.
The Goutorbe supplement and coefficient workbook currently remain internal.

Given a directory containing those four dataset directories:

```python
from pathlib import Path
from reskit.geothermal.preprocessing import create_geothermal_resource

inputs = Path("/path/to/Franzmann_GeothermalResourceInputs_2022")
resource = create_geothermal_resource(
    goutorbe_table=inputs / "goutorbe-global-heat-flow-2011/Supplementary material.txt",
    lithology_raster=inputs / "global-lithological-map-glim-v1/glim_wgs84_0point5deg.txt.asc",
    surface_temperature=inputs / "nasa-power-geothermal-climatology/POWER_Global_Climatology_LST_2.nc",
    conductivity_table=inputs / "geothermal-conductivity-assumptions/Mapper_Lithologic_to_heat_conductivity.xlsx",
    output_dir="geothermal-resource-regenerated",
)
```

Use `geothermal-resource-regenerated/Temperatures.nc4` as `sourceTemperature`
and `geothermal-resource-regenerated/heat_flow_sustainable_W_per_m2.nc4` as
`sourceSustainableHeatflow` in `egs_workflow`.

The function preserves the original water-body conductivity override, missing
NASA edge repair, 0.5-to-1-degree averaging and heat-production mask, including
the mask on the saved surface temperature. It validates grid coordinates and
units to prevent silent misalignment. The historical NASA climatology years
remain unknown. New metadata and floating-point differences in conductivity
averaging mean regenerated files are a new delivery; existing outputs are
refused rather than overwritten.
