# RESKit example: Northern Germany and the German North Sea

This example shows the full RESKit chain for four technologies in one region.

## Steps

1. **Build the simulation area.** The area has two parts:
   * Northern Germany: Schleswig-Holstein, Hamburg, Bremen, Niedersachsen and
     Mecklenburg-Vorpommern, read from the GADM level-1 shapefile.
   * The German North Sea EEZ, read from the Marine Regions World EEZ
     shapefile.

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

## Run the example

```bash
cd <this directory>/..
PROJ_DATA=/fast/home/p-dunkel/playground/RESKit_HA/.pixi/envs/default/share/proj \
  /fast/home/p-dunkel/playground/RESKit_HA/.pixi/envs/default/bin/python \
  example/northern_germany_north_sea.py
```

You can also run the cells one by one in an editor. Each `# %%` marker starts a
new cell.

## Configuration

Change these constants at the top of `northern_germany_north_sea.py`:

* `WEATHER_YEAR` - the ERA5 weather year.
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
