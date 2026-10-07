# RESKit example: Northern Germany and the German North Sea

This example shows the full RESKit chain for four technologies in one region.

## Steps

1. **Build the simulation area.** The area has two parts:
   * Northern Germany: Schleswig-Holstein, Hamburg, Bremen, Niedersachsen and
     Mecklenburg-Vorpommern, read from GADM 3.6 level 1.
   * The German North Sea EEZ, read from Marine Regions World EEZ v12.

   The German EEZ holds the North Sea and the Baltic Sea in one polygon. The
   example first removes all land from the EEZ. The Jutland peninsula splits
   the EEZ into the two seas then, and the example keeps the North Sea parts.
   The land removal also keeps the sea part and the land part of the area
   apart.
2. **Clip the placements.** The example reads the TREP-DB and geothermal
   placement databases and clips them to the area with `geopandas.clip`.
   It then takes a regular sample of `MAX_PLACEMENTS` placements.
3. **Simulate.** Each technology uses its RESKit workflow.
4. **Calculate the LCOE** of each placement.
5. **Plot** the LCOE of each placement on a map. The figure holds one map for
   each technology in a 2 x 2 grid.

## Technologies

| Technology | Workflow | CAPEX source |
| --- | --- | --- |
| Onshore wind | `reskit.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025` | `reskit.wind.onshore_turbine_capex` |
| Offshore wind | `reskit.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025` | `reskit.wind.calculate_specific_offshore_capex` |
| Openfield PV | `reskit.solar.openfield_pv_era5` | constant specific CAPEX |
| Geothermal EGS | `reskit.geothermal.egs_workflow` | the workflow gives the LCOE |

The offshore cost model needs the water depth and the distance to the coast of
each placement. The example reads both from rasters with
`reskit.util.local_values`.

RESKit has no PV cost model. The example uses the constant `PV_SPECIFIC_CAPEX`.

## Input data

The example takes every input from the ETHOS.Data catalogue. The collection
`example_northern_germany_north_sea` in `reskit/data/collections.yaml` names
each input once, and `reskit.data.paths()` returns the local path of each.

| Handle | Catalogue entry | Content |
| --- | --- | --- |
| `gadm_level1` | `gadm-3.6` (restricted) | GADM 3.6 level 1 |
| `eez` | `reskit-example-northern-germany-north-sea/german-eez-marine-regions-v12` | the German EEZ from Marine Regions World EEZ v12 |
| `onshore_wind_placements`, `offshore_wind_placements`, `openfield_pv_placements` | `trep-db` | TREP-DB 1.1.0 placements |
| `geothermal_placements` | `geothermal-egs-placements` | EGS placements, Franzmann et al. (2025) |
| `era5` | `era5-reskit-tiles-northern-germany-2018` (staged) | ERA5 2018, zoom-4 tiles x8/y4 and x8/y5 |
| `gwa_10m` ... `gwa_200m` | `global-wind-atlas-v4` | Global Wind Atlas 4.0 mean wind speed |
| `gsa_ghi`, `gsa_dni` | `global-solar-atlas-v2.9` | Global Solar Atlas 2.9 GHI and DNI |
| `water_depth` | `gebco-2025-combined` | GEBCO 2025 bathymetry |
| `coast_distance` | `dist2coast` | NASA distance to the nearest coast |

### One-time setup

1. Select the institute's internal catalogue. The public catalogue does not
   hold these datasets yet:

   ```bash
   reskit-data config set-catalog /fast/central/shared_data/ethos-data-catalog-internal/datacatalog.json
   ```

2. Stage the ERA5 tiles. They are not catalogued yet, so each machine registers
   them as development data:

   ```bash
   reskit-data staging add era5-reskit-tiles-northern-germany-2018 \
       /fast/central/shared_data/RESKit_example_northern_germany_north_sea/era5-reskit-tiles-northern-germany-2018
   ```

   The directory holds the tiles 4/8/4/2018 and 4/8/5/2018 of the processed
   ERA5 archive `ERA5_global_processed_V2022.02`. `reskit-data staging list`
   shows the entry.

3. GADM is restricted: its licence forbids redistribution, so `gadm-3.6` is
   never downloaded. It is read in place from the restricted cache on the ICE-2
   cluster. Outside the cluster, download `gadm36_levels_shp.zip` from
   <https://gadm.org/download_world36.html> under GADM's terms, unpack it, and
   register the copy in a restricted cache of your own:

   ```bash
   reskit-data config add-restricted-cache /path/to/my-restricted-cache
   ethos-data link gadm-3.6 /path/to/gadm36_levels_shp
   ```

## Run the example

```bash
cd <this directory>
python northern_germany_north_sea.py
```

You can also run the cells one by one in an editor. Each `# %%` marker starts a
new cell.

## Configuration

Change these constants at the top of `northern_germany_north_sea.py`:

* `WEATHER_YEAR` - the ERA5 weather year. The staged ERA5 tiles hold 2018 only.
* `MAX_PLACEMENTS` - the maximum number of placements for each technology.
  Set it to `None` to simulate all placements in the area. A value of 300 keeps
  the runtime at a few minutes.
* `NORTHERN_GERMAN_STATES` - the land part of the simulation area.
* `NORTH_SEA_MAX_LON` - the longitude that separates the North Sea part of the
  German EEZ from the Baltic Sea part.
* `ECONOMIC_ASSUMPTIONS`, `DISCOUNT_RATE`, `OFFSHORE_BASE_SPECIFIC_CAPEX` and
  `PV_SPECIFIC_CAPEX` - the economic assumptions.

## Output

The example writes to `output/`:

* `lcoe_maps.png` - the 2 x 2 grid of LCOE maps.
* `lcoe_<technology>.gpkg` - the LCOE of each placement as a GeoPackage.
