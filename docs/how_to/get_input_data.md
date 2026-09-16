# Get input data from the ETHOS.Data catalogue

RESKit reads its inputs from the shared ETHOS.Data catalogue and downloads them on demand into a cache that every ETHOS tool uses. At the end of this page the input files of a workflow are on your disk, first the small test data and then the full data, and you know where they are and how to check them. The [wind workflow example](../examples/3_wind/3_7_example_ethos_reskit_wind_workflow.ipynb) gets its inputs this way.

## Prerequisites

- RESKit is installed. In a development checkout run `pip install -e . --no-deps` once; this also installs the `reskit-data` command.
- ETHOS.Data is installed in the same environment. Until it is released on conda-forge and PyPI, install it from its repository with `pip install git+https://github.com/FZJ-IEK3-VSA/ETHOS.Data.git`; see the installation page of the [ETHOS.Data documentation](https://ethos-data.readthedocs.io/). `import reskit` works without it; `reskit.data` and `reskit-data` need it.
- Network access for the first download. Files already in the cache are not downloaded again.

## 1. Check what is in effect

```bash
reskit-data config show                                            # cache directory and catalogue in effect; works offline
reskit-data config set-catalog <path-or-url to datacatalog.json>   # institute users, once: read the internal catalogue
```

Until a public catalogue release contains the RESKit test data, the test collections resolve only against the internal catalogue. The setting is remembered for your account. If `reskit-data list` stops with `cannot read the catalogue index`, the catalogue RESKit pins is not reachable from your machine; set the catalogue as shown above.

## 2. See what RESKit needs

```bash
reskit-data list                        # every collection, one row per variant, with its size
reskit-data info onshore_wind --test    # the files and the named inputs of the test variant
reskit-data plan onshore_wind           # what a fetch of the full data would download
```

A collection holds the inputs of one workflow. `onshore_wind` has a `test` variant with small fixtures and a `full` variant with the real data; both name the same inputs (`era5`, `gwa_100m`, `gwa_50m`, `gwa_200m`), so the same code runs on either. The full data is the default; `--test` or `test=True` selects the fixtures.

## 3. Run an example on the test data

=== "Command line"

    ```bash
    reskit-data paths onshore_wind --test    # fetch the test data, print one line per input: name, tab, path
    ```

=== "Python"

    ```python
    import reskit as rk
    from reskit import data

    inputs = data.paths("onshore_wind", test=True)    # {input name: pathlib.Path}, fetched
    reskit_xr = rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(
        placements=placements,    # your turbine placements, prepared as in the wind workflow example
        era5_path=inputs["era5"],
        gwa_100m_path=inputs["gwa_100m"],
        height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
    )
    ```

## 4. Run on the full data

Drop the test flag. Run `reskit-data plan onshore_wind` first to see the download size.

```bash
reskit-data fetch onshore_wind    # download every file of the collection
reskit-data paths onshore_wind    # download and print the input paths
```

In Python, `data.paths("onshore_wind")` returns the same input names for the full data, and `data.fetch("onshore_wind")` returns every file as `{"<dataset>/<file>": Path}`.

The full ERA5 dataset is not catalogued yet. Until it is, `reskit-data list` marks `onshore_wind [full]` as `[unresolvable]` and the calls above fail for the full variant; the test variant works.

## 5. Get a single file or folder

Address data by its catalogue key, `<dataset>/<file>` or `<dataset>/<folder>`. `ls` shows what a dataset contains and fetches nothing; `path` fetches the file or folder if needed and prints its absolute path. Use `directory` where RESKit expects a folder of files, such as a weather source. A shapefile brings its sidecar files along.

=== "Command line"

    ```bash
    reskit-data ls reskit-test-data/global-wind-atlas
    reskit-data path reskit-test-data/global-wind-atlas/gwa100-like.tif    # one file
    reskit-data path reskit-test-data/era5                                 # a whole folder
    ```

=== "Python"

    ```python
    gwa_100m = data.path("reskit-test-data/global-wind-atlas/gwa100-like.tif")
    era5_dir = data.directory("reskit-test-data/era5")
    ```

## 6. Move the cache

The directory `config show` prints is the shared ETHOS.Data cache; `data.cache_dir()` returns it in Python. To keep the data on another disk for this Python environment:

```bash
reskit-data config set-cache <dir> --scope environment
```

Scopes are `user` (default), `project`, `environment` and `site`. For one shell or job, set `ETHOS_DATA_DIR` instead; `--root <dir>` before a subcommand does the same for one command. Existing files are not moved; the next fetch downloads into the new directory.

## Check that it worked

```bash
reskit-data verify onshore_wind --test --deep
```

`verify` compares the files on disk with the catalogue, by size, or by SHA-256 checksum with `--deep`. It prints a count per status, ends with `N file(s) match the catalogue.` and exits with 0. If a file does not match, `reskit-data verify onshore_wind --test --repair` fetches it again; add `--dry-run` to see first what would be re-fetched. After step 3, `reskit-data paths onshore_wind --test` prints four lines, and `inputs["era5"]` in Python is an existing directory.

## Go further

The [ETHOS.Data documentation](https://ethos-data.readthedocs.io/) covers what this page only touches:

- the shared cache and the internal catalogue, and how to configure them for a machine, a project or a CI job
- restricted data that may not be downloaded, and how to use a copy that is already on disk
- the `ethos-data` command reference, of which `reskit-data` is the RESKit-specific front end
