# Get input data from the ETHOS.Data catalogue

Use `reskit.data` or `reskit-data` for the collections RESKit ships. They share
one catalogue selection and the ETHOS cache. The
[wind workflow example](../examples/3_wind/3_7_example_ethos_reskit_wind_workflow.ipynb)
uses this interface.

Install RESKit and ETHOS.Data in the same environment. In a development checkout,
`pip install -e . --no-deps` installs the `reskit-data` console script.
See [ETHOS.Data installation](https://ethos-data.readthedocs.io/en/latest/installation/)
for the shared dependency. Catalogue metadata and uncached public files need
network access unless local copies are selected.

## Select the catalogue

```bash
reskit-data config show
reskit-data list
```

`config show` reports shared configuration and origins offline. `list` prints
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

## Get the inputs a workflow needs

```bash
reskit-data info onshore_wind --test
reskit-data plan onshore_wind --test
reskit-data paths onshore_wind --test
```

`info` lists the selected files and named inputs; `plan` previews transfers;
`paths` fetches them and prints one `handle<TAB>absolute path` line per input.

In Python, pass those named inputs to the workflow:

```python
import reskit as rk
from reskit import data

inputs = data.paths("onshore_wind", test=True)
result = rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(
    placements=placements,  # prepared as in the wind workflow example
    era5_path=inputs["era5"],
    gwa_100m_path=inputs["gwa_100m"],
    height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
)
```

Both variants offer the same input names. Preview the full selection before
dropping `--test` or `test=True`:

```bash
reskit-data plan onshore_wind
reskit-data fetch onshore_wind
reskit-data paths onshore_wind
```

`fetch` makes the whole collection available; `paths` also returns its named
inputs. Full data is the default. An `[unresolvable]` row means the selected
catalogue or collection definition needs attention before that variant can run.

## Access a catalogue key

```bash
reskit-data ls reskit-test-data/era5
reskit-data path reskit-test-data/era5
```

`ls` reads metadata only. `path` fetches a file, folder, dataset or family and
prints its local path; shapefiles include sidecars. In Python:

```python
era5_dir = data.directory("reskit-test-data/era5")
gwa_100m = data.path("reskit-test-data/global-wind-atlas/gwa100-like.tif")
```

These use RESKit's selected catalogue. For direct access through `ethos-data`,
the corresponding retrieval command is `ethos-data fetch KEY`; explicitly
choose the same catalogue when comparing results.

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
reskit-data paths trial_weather
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
reskit-data verify onshore_wind --test --deep
```

Expect matching files and exit status `0`. For a damaged downloaded copy,
preview with `reskit-data verify onshore_wind --test --deep --repair --dry-run`,
then remove `--dry-run` to repair. In-place and restricted data need correction
at their source; staged files are unverifiable.

Detailed procedures have one home in ETHOS.Data:

- [Configuration and cache locations](https://ethos-data.readthedocs.io/en/latest/how-to/set-up-your-machine/).
- [Integrity checking and repair](https://ethos-data.readthedocs.io/en/latest/how-to/verify-and-repair/).
- [Exporting and refreshing repository test bundles](https://ethos-data.readthedocs.io/en/latest/how-to/keep-test-data-in-a-repository/).
- [Package command options](https://ethos-data.readthedocs.io/en/latest/reference/cli/package-data/).

Shared cache administration uses `ethos-data link`, `unlink` and `materialize`;
catalogue publishing uses `ethos-data catalog`.
