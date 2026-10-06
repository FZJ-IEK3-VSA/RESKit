# Check the inputs of a workflow

A wrong input, such as a missing placement column, a weather source without a variable
the workflow needs, or a raster path with a typo, would otherwise show up only during
the simulation, possibly after hours. RESKit checks the inputs before it simulates. The
check reads only metadata and the raster values at the placements, so it takes seconds,
also for large inputs.

## What is checked

- **Arguments**: they match the signature of the workflow.
- **Placements**: valid locations (point geometries, or `lon` and `lat` in range) and the
  columns the workflow needs, e.g. `capacity`, `hub_height`, and `rotor_diam` or
  `powerCurve` for wind. Required numeric columns must have a value for every placement.
- **Weather sources**: each source opens and provides the variables the workflow reads,
  with the time steps of its time axis. It covers every placement and has a regular time
  axis. A `time_slice` lies within the available time span, and a `https://` Zarr store
  has credentials in `~/.netrc`.
- **Input files**: they exist, and every raster has a value at every placement.
- **Output files**: their directory exists and is writable.

## Check before a run

Every workflow checks its inputs itself before it simulates. If it finds an error, it
raises a `ResError` listing all problems at once. It emits each warning as a Python
warning and runs on.

To check without running the simulation, pass the same arguments to
`rk.validate_inputs`:

```python
import reskit as rk

report = rk.validate_inputs(
    "wind_era5_PenaSanchezDunkelWinklerEtAl2025",  # or the workflow function itself
    placements,
    era5_path=inputs["era5"],
    gwa_100m_path=inputs["gwa_100m"],
    height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
)
print(report)
report.raise_if_errors()  # raises a ResError if an error was found
```

`report.ok` is `False` if an error was found, and `report.errors` and `report.warnings`
list the findings.

## Read the report

```text
Inputs of wind_era5_PenaSanchezDunkelWinklerEtAl2025: 3 error(s), 2 warning(s)
  ERROR   [placements] placements need the column 'hub_height'
  INFO    [weather 'era5_path'] 140 time steps from 2014-12-31 23:30:00 to 2015-01-06 18:30:00
  ERROR   [weather 'era5_path'] does not cover 1 of 2 placements, e.g. lon=20.0, lat=60.0
  ERROR   [gwa_100m_path] does not exist: /data/gwa_100m.tif
  WARNING [height_scaling_data[50]] has no value at 1 of 2 placements (outside the raster or nodata)
  WARNING [height_scaling_data[200]] has no value at 1 of 2 placements (outside the raster or nodata)
```

Each line names the input in brackets: `placements`, a weather source by its argument,
or a file argument, with the key or index for a dict or list of files.

- `ERROR`: the workflow would fail, or give wrong results.
- `WARNING`: the workflow runs, but probably not as intended. For example, a placement
  outside a raster gets the workflow's fallback value, or NaN where there is none.
- `INFO`: for your information, e.g. the time span that will be simulated.

## Skip the check

Pass `validate=False` to a workflow, e.g. if you checked the inputs already, or if the
check rejects an input that you know to be fine. In the latter case, please report it as
a bug.

```python
rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(placements, ..., validate=False)
```

## Declare the inputs of a new workflow

The check knows what a workflow needs because every workflow declares its inputs with
`@declare_inputs`:

```python
from reskit.util.input_validation import WeatherInput, declare_inputs


@declare_inputs(
    WindWorkflowManager,
    weather={
        "era5_path": WeatherInput(
            "ERA5",
            ("elevated_wind_speed", "surface_pressure", "surface_air_temperature", "boundary_layer_height"),
        )
    },
    files=("gwa_100m_path", "height_scaling_data"),
)
def wind_era5_PenaSanchezDunkelWinklerEtAl2025(placements, era5_path, gwa_100m_path, height_scaling_data, ...):
    wf = WindWorkflowManager(placements)
    wf.read(
        variables=["elevated_wind_speed", "surface_pressure", "surface_air_temperature", "boundary_layer_height"],
        source_type="ERA5",
        source=era5_path,
        ...
    )
```

- **The workflow manager** checks the placements with its `placement_problems()`. A new
  workflow manager which needs further columns extends it, and its constructor then
  rejects placements without them:

  ```python
  @classmethod
  def placement_problems(cls, placements):
      return super().placement_problems(placements) + _numeric_column_problems(placements, "capacity")
  ```

- **`weather`** maps each argument which gives the path of a weather source to a
  `WeatherInput`. A `WeatherInput` holds the source type and the variables the workflow
  passes to `read()`, and the `time_index_from` if `read()` gets one. If the caller
  chooses the source type, name the argument which gives it with `source_type_argument`
  instead of a source type.
- **`files`** names the arguments which may give input files. Such an argument may also
  give a dict or a list of files, or a value which is not a path, e.g. a number; the check
  skips the latter.
- **`outputs`** names the arguments which may give an output file or directory. By default,
  this is `output_netcdf_path`.

`test/05_workflows/test_input_validation.py` checks that the declared weather inputs
match the `read()` calls of every workflow, so a declaration cannot go out of date
unnoticed.
