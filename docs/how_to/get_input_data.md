# Get input data from the ETHOS.Data catalogue

Use `reskit.data` or `reskit-data` for the collections RESKit ships. They share
one catalogue selection and the ETHOS cache. The
[wind workflow example](../examples/3_wind/3_7_example_ethos_reskit_wind_workflow.ipynb)
uses this interface.

Install RESKit and ETHOS.Data in the same environment. In a development checkout,
`pip install -e . --no-deps` installs the `reskit-data` console script.
See [ETHOS.Data installation](https://ethos-data.readthedocs.io/en/latest/installation/)
for the shared dependency. The `reskit-test-data` fixtures ship with RESKit as a
verified bundle and are read from it by default; every other dataset needs network
access for catalogue metadata and uncached files.

## Get the inputs a workflow needs

Call `data.paths()` immediately before the workflow. It returns the local input
paths, using the bundled fixtures or the shared cache and fetching missing
catalogued data as needed. A workflow's
collection has the same name as its Python function:

```python
import reskit as rk
from reskit import data

inputs = data.paths("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True)
result = rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(
    placements=placements,  # prepared as in the wind workflow example
    era5_path=inputs["era5"],
    gwa_100m_path=inputs["gwa_100m"],
    height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
)
```

Here `test=True` uses the small fixtures shipped with RESKit, so input resolution
works offline. Both variants offer the same input names, and omitting `test=True`
selects the full variant. **The pinned public catalogue currently has no full
ERA5 dataset**, so this workflow's full variant raises `UnknownDataset`. Select
a catalogue containing the required full inputs before using it; removing the
flag alone is not enough.

### Optional command-line access

The CLI is useful for inspecting inputs, planning a download or filling a cache
before running Python:

```bash
reskit-data show wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test
reskit-data fetch wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --plan
reskit-data fetch wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --paths
```

`show` describes the collection and its named inputs and downloads nothing.
`fetch` transfers data; `--plan` previews the transfer instead of running it.
`--paths` prints one `handle<TAB>absolute path` line per input once the files are
there. Add `--files` to `show` for the full file list.

The CLI uses the catalogue and shared cache even for test fixtures; the Python
example above reads the bundled copy by default. To preview the full variant:

```bash
reskit-data fetch wind_era5_PenaSanchezDunkelWinklerEtAl2025 --plan
```

An `[unresolvable]` row means the selected catalogue or collection definition
needs attention before that variant can run.

## The bundled test fixtures

RESKit carries its copy of the `reskit-test-data` family in `reskit/data/test_cache`
as an [ETHOS.Data bundle](https://ethos-data.readthedocs.io/en/latest/how-to/keep-test-data-in-a-repository/):
`bundle.json` records every file with the size and SHA-256 the pinned catalogue
declares, the files sit under `data/<dataset>/<path>`, and `datasets/` archives
the licences. `paths`, `fetch`, `path` and `directory` answer from that copy
whenever it holds what was asked for, checked against those hashes once per
process and never downloaded, so the examples and the test suite run offline.
Everything the bundle does not hold comes from the catalogue as described below.

To fetch the fixtures from the catalogue's store instead, into the shared cache
like any other dataset, pass `download=True` or set `RESKIT_DATA_DOWNLOAD=1`:

```python
era5_dir = data.directory("reskit-test-data/era5", download=True)
inputs = data.paths("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True, download=True)
```

=== "Bash"

    ```bash
    export RESKIT_DATA_DOWNLOAD=1
    ```

=== "PowerShell"

    ```powershell
    $env:RESKIT_DATA_DOWNLOAD = "1"
    ```

The argument wins over the variable. A bundled file that is missing or altered
is an error, never a reason to download: the copy in a checkout is what the
tests run on. `reskit-data fetch`, `show` and `verify` always work through the
catalogue, as do `reskit.data.plan()` and `describe()`; `reskit-data bundle`
works on the copy:

```bash
reskit-data bundle verify reskit/data/test_cache test_suite   # the bundled copy
reskit-data fetch test_suite --plan                            # what the catalogue route would transfer
```

### Refresh the bundle

Regenerate the bundle whenever the catalogue pin in `collections.yaml` moves or
a fixture changes. Export runs against the catalogue named with `--catalog` (pass
the pin explicitly, so a machine-wide catalogue setting cannot leak into the
manifest), takes the existing files as verified input, and refuses a target that
exists; so export beside the bundle and move the manifest over:

=== "Bash"

    ```bash
    pin=$(sed -n 's/^catalog: *//p' reskit/data/collections.yaml)
    roots=""
    for m in $(ls reskit/data/test_cache/data/reskit-test-data); do
      roots="$roots --source-root reskit-test-data/$m=reskit/data/test_cache/data/reskit-test-data/$m"
    done
    reskit-data --catalog "$pin" bundle export reskit/data/test_cache-next test_suite test_suite_public \
      --source-revision "$pin" $roots
    mv reskit/data/test_cache-next/bundle.json reskit/data/test_cache/
    rm -rf reskit/data/test_cache/datasets && mv reskit/data/test_cache-next/datasets reskit/data/test_cache/
    rm -rf reskit/data/test_cache-next
    reskit-data bundle verify reskit/data/test_cache test_suite
    ```

=== "PowerShell"

    ```powershell
    $pin = (Select-String '^catalog:\s*(\S+)' reskit/data/collections.yaml).Matches[0].Groups[1].Value
    $roots = Get-ChildItem reskit/data/test_cache/data/reskit-test-data -Directory |
      ForEach-Object { "--source-root"; "reskit-test-data/$($_.Name)=reskit/data/test_cache/data/reskit-test-data/$($_.Name)" }
    reskit-data --catalog $pin bundle export reskit/data/test_cache-next test_suite test_suite_public `
      --source-revision $pin @roots
    Move-Item reskit/data/test_cache-next/bundle.json reskit/data/test_cache/ -Force
    Remove-Item reskit/data/test_cache/datasets -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item reskit/data/test_cache-next/datasets reskit/data/test_cache/
    Remove-Item reskit/data/test_cache-next -Recurse -Force
    reskit-data bundle verify reskit/data/test_cache test_suite
    ```

Export fails if a file in the repository differs from the catalogue, or if the
catalogue's own metadata is inconsistent, for instance a licence document that
does not match its recorded hash. Fix the source; never edit the manifest.
Commit `bundle.json`, `datasets/` and any changed fixture together with the pin.
`--source-revision` is a provenance label; the pin selects the revision.

## Select the catalogue

```bash
reskit-data config show
reskit-data show
```

`config show` reports shared configuration and origins offline. `show` prints
the actual catalogue selected by RESKit and its collections. RESKit uses, in order:
`--catalog`, `RESKIT_DATA_CATALOG`, shared environment/configuration, then the pin
in `reskit/data/collections.yaml`.

For a RESKit-specific override:

=== "Bash"

    ```bash
    export RESKIT_DATA_CATALOG=/path/to/datacatalog.json
    ```

=== "PowerShell"

    ```powershell
    $env:RESKIT_DATA_CATALOG = "D:/catalogue/datacatalog.json"
    ```

For one invocation, put `--catalog LOCATION` before the subcommand. To configure
all ETHOS packages, follow
[shared machine setup](https://ethos-data.readthedocs.io/en/latest/how-to/set-up-your-machine/).
If the pin cannot be read or lacks an input, select a complete catalogue version
provided by the maintainer.

## Access a catalogue key

`reskit-data` works in collections. A single dataset, folder or file is
`ethos-data`'s to hand out:

```bash
ethos-data ls reskit-test-data/era5
ethos-data fetch reskit-test-data/era5
```

`ls` reads metadata only. `fetch` retrieves a file, folder, dataset or family
and prints its local path; shapefiles include sidecars. In Python, through
RESKit's own catalogue selection:

```python
era5_dir = data.directory("reskit-test-data/era5")
gwa_100m = data.path("reskit-test-data/global-wind-atlas/gwa100-like.tif")
```

The Python calls use RESKit's selected catalogue; `ethos-data` uses the shared
settings, so pass the same `--catalog` when comparing results.

## Location rasters and the turbine library

Three inputs used to be configured in `reskit/default_paths.yaml`, a file inside
the installed package. They are now arguments, with defaults that come from the
catalogue:

| Input | Function | Default | Collection |
| --- | --- | --- | --- |
| Water depth raster (GEBCO) | `water_depth_from_location(..., waterDepthFilePath=)` | the `water_depth` handle of `offshore_siting`, fetched on first use | `offshore_siting` |
| Distance-to-coast raster | `distance_to_coastline(..., distancetoCoastFilePath=)` | the `coast_distance` handle of `offshore_siting`, fetched on first use | `offshore_siting` |
| Turbine library | `rk.wind.turbine_library(path=)` | the 124 turbines RESKit ships | `turbine_library` (licensed) |

Baseline turbine definitions are `OnshoreParameters(fp=...)` and
`OffshoreParameters(fp=...)`; RESKit ships the defaults and no dataset is involved.

```python
import reskit as rk
from reskit import data
from reskit.util.local_values import distance_to_coastline, water_depth_from_location

inputs = data.paths("offshore_siting", test=True)  # the German Bight fixtures
depth = water_depth_from_location(54.4, 6.9, waterDepthFilePath=inputs["water_depth"])
distance = distance_to_coastline(54.4, 6.9, distancetoCoastFilePath=inputs["coast_distance"])

water_depth_from_location(54.4, 6.9)  # no path: the full grids, from the catalogue

rk.wind.turbine_library(data.paths("turbine_library")["turbines"])  # the licensed library
rk.wind.turbine_library("/path/to/my/turbines")  # or any directory of turbine CSVs
```

A directory given to `turbine_library` becomes the library for the rest of the
process, so workflows that name a power curve resolve it there too. The
`turbine_library` collection is restricted data, read from the restricted cache
and never downloaded; outside the institute the call raises an access error and
the shipped library stays in use.

To read a private copy of a catalogued dataset, for instance GEBCO tiles already
on your disk, tell ETHOS.Data where it lies instead of passing the path to every
call:

```bash
ethos-data config set-root gebco-2024 /data/gebco
```

Until the maintainers catalogue GEBCO and publish `dist2coast`, the full variant
of `offshore_siting` is `[unresolvable]` and the two functions need an explicit
path; `reskit-data show offshore_siting` reports the state, and the
`TODO(maintainer)` note in `collections.yaml` says what to change.

## Develop against unpublished data

In a development checkout, put a candidate in its own directory and register it:

```bash
reskit-data config set-staging-cache /scratch/me/ethos-staging
reskit-data staging add trial-weather /scratch/me/candidate --copy --note "local experiment"
reskit-data staging list
```

Add a collection such as `trial_weather` to RESKit's shipped
`reskit/data/collections.yaml`, selecting the staged dataset:

```yaml
  trial_weather:
    include:
      - dataset: trial-weather
    paths:
      weather: trial-weather
```

Place that entry under the existing `collections:` mapping, then use:

```bash
reskit-data fetch trial_weather --paths
```

The staged directory supplies the files and produces a warning. A copied staging
entry is a snapshot; source edits need restaging. The root is shared across ETHOS
packages, and restricted datasets are never shadowed. Registration works offline;
collection resolution still needs a readable catalogue index.

After the candidate is accepted and RESKit's pin/selection is updated:

```bash
reskit-data staging remove trial-weather --force
```

This deletes the staged copy and leaves the source directory intact. Remove
`--force` for linked entries. The
[development/proposal guide](https://ethos-data.readthedocs.io/en/latest/how-to/propose-a-dataset/)
covers review and adoption; the
[staging reference](https://ethos-data.readthedocs.io/en/latest/reference/cli/package-data/#staging)
lists all options.

## Check the result

```bash
reskit-data verify wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --deep
```

Expect matching files and exit status `0`. For a damaged downloaded copy,
preview with `reskit-data verify wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --deep --repair --dry-run`,
then remove `--dry-run` to repair. In-place and restricted data need correction
at their source; staged files are unverifiable.

Detailed procedures have one home in ETHOS.Data:

- [Configuration and cache locations](https://ethos-data.readthedocs.io/en/latest/how-to/set-up-your-machine/).
- [Integrity checking and repair](https://ethos-data.readthedocs.io/en/latest/how-to/verify-and-repair/).
- [Exporting and refreshing repository test bundles](https://ethos-data.readthedocs.io/en/latest/how-to/keep-test-data-in-a-repository/).
- [Package command options](https://ethos-data.readthedocs.io/en/latest/reference/cli/package-data/).

Shared cache administration uses `ethos-data link`, `unlink` and `materialize`;
catalogue publishing uses `ethos-data catalog`.
