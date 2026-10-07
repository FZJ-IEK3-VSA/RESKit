# Get input data from the ETHOS.Data catalogue

Use `reskit.data` or `reskit-data` for the collections RESKit ships. They share
one catalogue selection and the ETHOS cache. The
[wind workflow example](../examples/3_wind/3_7_example_ethos_reskit_wind_workflow.ipynb)
uses this interface.

Install RESKit and ETHOS.Data in the same environment. In a development checkout,
`pip install -e . --no-deps` installs the `reskit-data` console script.
See [ETHOS.Data installation](https://ethos-data.readthedocs.io/en/latest/installation/)
for the shared dependency. The `reskit-test-data` fixtures ship with RESKit as a
verified bundle and are read from it, offline; every other dataset needs network
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
selects the full variant. **No catalogue holds a full ERA5 dataset yet**, so this
workflow's full variant raises `UnknownDataset`. The workflows whose full inputs
are not catalogued at all -- the MERRA-2, SARAH and other ERA5 workflows -- have a
`test` variant only, and asking for their full variant raises a `CollectionError`
that says so.

The examples take their placements from the `example_placements` collection:

```python
import pandas as pd

placements = pd.read_csv(data.paths("example_placements")["turbines"])
```

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
there. Add `--files` to `show` for the full file list. `show` and `fetch --paths`
answer a test variant from the bundled copy, like the Python call; `fetch --plan`
and `verify` read the catalogue. To preview the full variant:

```bash
reskit-data fetch wind_era5_PenaSanchezDunkelWinklerEtAl2025 --plan
```

An `[unresolvable]` row in `reskit-data show` means the selected catalogue or the
collection definition needs attention before that variant can run.

## The bundled test fixtures

RESKit carries the `reskit-test-data` family in `reskit/data/test_cache` as an
[ETHOS.Data bundle](https://ethos-data.readthedocs.io/en/latest/how-to/package-maintainers/keep-data-in-the-repository/):

| Path | Content |
| --- | --- |
| `bundle.json` | each dataset's alignment with the catalogue, and every file's size and SHA-256 |
| `data/reskit-test-data/<member>/` | the fixtures |
| `datasets/reskit-test-data/<member>/` | each member's description, `dataset.yaml`, and its licence documents |

`reskit.data` lists the bundle when it builds its ETHOS.Data handle, so ETHOS.Data
reads it before the catalogue: the files are checked against `bundle.json` once per
process, never downloaded, and a collection the bundle holds whole reads no
catalogue index at all. That is what lets the examples' test variants and the test
suite run offline. The tests name their fixtures through the `test_suite`
collection:

```python
from reskit import data

FIXTURES = data.paths("test_suite")
source = rk.weather.Era5Source(FIXTURES["era5"])
```

A bundled file that is missing, or changed without `reskit-data bundle update`
recording it, is an error, never a reason to download. To read the fixtures
through the catalogue's store and the shared cache instead, for example to test
the download, set the switch every ETHOS.Data package honours:

=== "Bash"

    ```bash
    export ETHOS_DATA_DOWNLOAD=1
    ```

=== "PowerShell"

    ```powershell
    $env:ETHOS_DATA_DOWNLOAD = "1"
    ```

Check the bundle offline:

```bash
reskit-data bundle verify reskit/data/test_cache
```

It reports every file, description and licence document as `ok`, `modified`,
`missing` or `unrecorded`, and each dataset's alignment.

The bundle is currently ahead of the catalogue in `reskit-test-data/placements`:
it holds the two `_cityBulawayoInZimbabwa_2025` tables of the ICON-LAM regression
tests, which the catalogue does not hold yet, so every process that reads the
bundle warns once with `ethos_data.BundleAlignmentWarning` until the catalogue
takes them in (see [Change a fixture](#change-a-fixture)).

### Change a fixture

Edit the files under `data/`, then record the change:

```bash
reskit-data bundle update reskit/data/test_cache
```

The bundle is then ahead of the catalogue: it is read as recorded, and every
process that reads it warns once with `ethos_data.BundleAlignmentWarning` until it
is realigned. Realign it one of two ways:

- **The catalogue takes the bundle's version.** `reskit-data propose
  reskit/data/test_cache` drafts the proposal for the datasets that are ahead; the
  catalogue maintainers take it in with `ethos-data catalog add-bundle` and release.
  Then run `reskit-data bundle update reskit/data/test_cache` with that catalogue
  selected to record the new alignment.
- **The bundle takes the catalogue's version**, to drop a change or to catch up
  with a later revision of a member:

  ```bash
  reskit-data bundle update reskit/data/test_cache --from-catalog reskit-test-data/era5
  ```

A bundle holds public data with settled licensing only. Commit `bundle.json`,
`data/` and `datasets/` together. `.gitattributes` forbids line-ending conversion
for the whole bundle, since a converted file no longer matches its SHA-256.

## Select the catalogue

```bash
reskit-data config show
reskit-data show
```

`config show` reports shared configuration and origins offline. `show` prints
the actual catalogue selected by RESKit and its collections. RESKit uses, in order:
`--catalog`, `RESKIT_DATA_CATALOG`, then `ETHOS_DATA_CATALOG` or the catalogue in
the shared settings file, then the public catalogue.

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
[shared machine setup](https://ethos-data.readthedocs.io/en/latest/how-to/data-users/set-up-your-machine/).
Inside ICE-2, select the institute's internal catalogue: several datasets the
examples read are not in the public one yet.

`reskit/data/collections.yaml` does not bound the catalogue release yet. During
the beta no catalogue records a release, and ETHOS.Data refuses any catalogue that
records none once a collections file declares bounds; the file says what to add
with the first release.

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
gwa_100m = data.catalog_path("global-wind-atlas-v4/wind_speed_cog_100m.tif")
```

`catalog_path` reads the catalogue index even for a key the bundle holds, so it is
for trying a dataset before a collection names it; workflows, examples and tests
take their inputs from `data.paths()`. The Python call uses RESKit's selected
catalogue; `ethos-data` uses the shared settings, so pass the same `--catalog`
when comparing results.

## Location rasters and the turbine library

Three inputs used to be configured in `reskit/default_paths.yaml`, a file inside
the installed package. They are now arguments, with defaults that come from the
catalogue:

| Input | Function | Default | Collection |
| --- | --- | --- | --- |
| Water depth raster (GEBCO 2025) | `water_depth_from_location(..., waterDepthFilePath=)` | the `water_depth` handle of `offshore_siting`, fetched on first use | `offshore_siting` |
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

The full variant of `offshore_siting` is GEBCO 2025 as one global raster
(`gebco-2025-combined`) and NASA's distance-to-coast grid (`dist2coast`), about
5 GB together.

A directory given to `turbine_library` becomes the library for the rest of the
process, so workflows that name a power curve resolve it there too. The
`turbine_library` collection is restricted data, read from a restricted cache and
never downloaded; outside the institute the call raises an access error that says
how to obtain it, and the shipped library stays in use.

To use a copy of a catalogued dataset that is already on your disk, for instance
the GEBCO raster, register it with ETHOS.Data instead of passing the path to every
call. `link` makes the dataset's cache entry a link to the copy, which is then
read in place; `reskit-data verify offshore_siting --deep` checks it against the
catalogue:

```bash
ethos-data link gebco-2025-combined /data/gebco-2025-combined
```

A restricted dataset is registered the same way, in a restricted cache your
settings list:

```bash
reskit-data config add-restricted-cache /path/to/my-restricted-cache
ethos-data link reskit-turbine-library /path/to/turbines
```

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

Staging is also how the examples that need ERA5 run until ERA5 is catalogued: the
long-run-average example and the `example_north_sea_offshore_wind` collection read
the processed ERA5 archive as the dataset `era5`, and
`example_northern_germany_north_sea` reads two of its tiles as
`era5-reskit-tiles-northern-germany-2018`. On a machine that holds the archive:

```bash
reskit-data staging add era5 /path/to/ERA5_global_processed_V2022.02
```

After a candidate is accepted into the catalogue:

```bash
reskit-data staging remove trial-weather --force
```

This deletes the staged copy and leaves the source directory intact. Remove
`--force` for linked entries. The
[development/proposal guide](https://ethos-data.readthedocs.io/en/latest/how-to/package-maintainers/propose-a-dataset/)
covers review and adoption; the
[staging reference](https://ethos-data.readthedocs.io/en/latest/reference/cli/package-data/#staging)
lists all options.

## Check the result

```bash
reskit-data verify wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --deep
```

Expect matching files and exit status `0`. For a damaged downloaded copy,
preview with `reskit-data verify <collection> --deep --repair --dry-run`,
then remove `--dry-run` to repair. In-place and restricted data need correction
at their source; staged files are unverifiable. The bundled fixtures are checked
with `reskit-data bundle verify reskit/data/test_cache`.

Detailed procedures have one home in ETHOS.Data:

- [Configuration and cache locations](https://ethos-data.readthedocs.io/en/latest/how-to/data-users/set-up-your-machine/).
- [Integrity checking and repair](https://ethos-data.readthedocs.io/en/latest/how-to/data-users/verify-and-repair/).
- [Keeping data in the repository](https://ethos-data.readthedocs.io/en/latest/how-to/package-maintainers/keep-data-in-the-repository/).
- [Package command options](https://ethos-data.readthedocs.io/en/latest/reference/cli/package-data/).

Shared cache administration uses `ethos-data link`, `unlink` and `materialize`;
catalogue publishing uses `ethos-data catalog`.
