"""Create the Zarr twins of the ERA5 netCDF4 test fixtures.

RESKit reads ERA5 either from a directory of netCDF4 files (Era5Source) or from a
Zarr store (Era5ZarrSource). So that the test suite can run every ERA5 test against
both, each netCDF4 fixture set is converted into one Zarr store holding exactly the
same data:

    era5/*.nc       -> era5-zarr/era5.zarr
    era5-csp/*.nc   -> era5-zarr/era5-csp.zarr

The stores sit in a directory of their own, so that they are one member of the
ETHOS.Data catalogue rather than part of 'era5' and 'era5-csp': whoever downloads
those members, e.g. through a collection of reskit/data/collections.yaml, keeps
getting the netCDF4 files only. The single-cell file of era5-csp has no Zarr twin,
as no test reads it.

The stores are derived from the netCDF4 files rather than downloaded again from an
online Zarr store (e.g. the Earth Data Hub), because the tests compare against
values computed from the netCDF4 fixtures: those were cut from a 2020 CDS download,
are packed to 16 bit integers and partly preprocessed by RESKit (ws100, *_t_adj, ...)
-- none of which an online store reproduces bit for bit.

What the conversion preserves, and why:

* The values, bit for bit. Each variable keeps the packing of its netCDF4 file
  (integer dtype, scale_factor, add_offset, _FillValue), so decoding yields the
  same numbers, and so does the time encoding.
* The time alignment as RESKit sees it. Era5Source reads every file positionally
  and takes the time axis from a single file -- the one holding 'time_index_from',
  if given -- it never aligns files by their own time coordinates. A Zarr store has
  one time axis, so where the files disagree it is taken from the variable which
  RESKit reads them with ('time_axis_from' in STORES), and every other variable is
  put on it by position, exactly as Era5Source reads it. This concerns era5-csp: its
  '*_t_adjusted' radiation is labelled one hour later than the other two files (the
  shift of its CDO preprocessing), and csp_ptr_era5 reads the set with
  time_index_from="direct_horizontal_irradiance". The original first timestamp of a
  relabelled variable is kept in its attributes as 'reskit_original_time_start'.
* The dimension names of the netCDF4 files ('time', 'latitude', 'longitude').

The stores are written as Zarr format 3 with consolidated metadata, which
Era5ZarrSource reads by default. The output is deterministic -- one chunk per
variable, a fixed compressor, no timestamps in the metadata -- so rerunning the
script reproduces the committed stores byte for byte, which `--check` verifies. That
holds for the zarr version they were written with (zarr 3.1.5): other versions or
Blosc builds may lay out the metadata or compress larger chunks differently, while
the data stays identical, which test/02_weather_source/test_Era5ZarrTestData.py
checks in every environment.

Usage (from the repository root):

    python scripts/make_era5_zarr_test_data.py           # (re)write the stores
    python scripts/make_era5_zarr_test_data.py --check   # verify the committed stores
"""

import argparse
import filecmp
import hashlib
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec

FIXTURES = Path(__file__).resolve().parents[1] / "reskit" / "data" / "test_cache" / "data" / "reskit-test-data"

# Each store, relative to FIXTURES: the netCDF4 files it is made from, and the variable
# whose time axis it takes where the files disagree (None: they have to agree).
STORES = {
    "era5-zarr/era5.zarr": ("era5/*.nc", None),
    "era5-zarr/era5-csp.zarr": ("era5-csp/*.nc", "fdir_t_adj"),
}

TIME = "time"

# Shared by every variable, so the stores do not depend on the defaults of the
# installed zarr version.
COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")

# The encoding keys which carry over from netCDF4 to Zarr. All others describe the
# netCDF4/HDF5 storage layout (zlib, chunksizes, contiguous, ...) and are dropped.
KEPT_ENCODING = ("dtype", "scale_factor", "add_offset", "_FillValue", "missing_value", "units", "calendar")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encoding(variable: xr.Variable) -> dict:
    return {key: variable.encoding[key] for key in KEPT_ENCODING if key in variable.encoding}


def build_dataset(files: list[Path], time_axis_from: str | None = None) -> xr.Dataset:
    """Merge the netCDF4 files of one fixture set into a single dataset.

    Parameters
    ----------
    files : list of Path
        The netCDF4 files, all on the same latitude/longitude grid and with the same
        number of time steps
    time_axis_from : str, optional
        The variable whose time axis the dataset takes. Only needed, and only allowed,
        if the files do not all share the same time axis.

    Returns
    -------
    xarray.Dataset
        One data variable per variable of the files, all on the same time axis by
        position, with the netCDF4 packing kept in each variable's encoding
    """
    datasets = [xr.open_dataset(path, decode_times=True, mask_and_scale=True) for path in files]
    try:
        axes = {tuple(ds[TIME].values) for ds in datasets}
        if len({len(axis) for axis in axes}) != 1:
            raise ValueError(f"The files do not have the same number of time steps: {[str(f) for f in files]}")
        if len(axes) > 1 and time_axis_from is None:
            raise ValueError(f"The files have different time axes, name the one to use: {[str(f) for f in files]}")
        if len(axes) == 1 and time_axis_from is not None:
            raise ValueError(
                f"The files share one time axis, 'time_axis_from' is not needed: {[str(f) for f in files]}"
            )
        reference = next(
            (ds for ds in datasets if time_axis_from is None or time_axis_from in ds.data_vars),
            None,
        )
        if reference is None:
            raise ValueError(f"No file holds 'time_axis_from' variable {time_axis_from!r}: {[str(f) for f in files]}")
        shared_axis = tuple(reference[TIME].values)

        merged = xr.Dataset(coords={name: reference[name].copy() for name in (TIME, "latitude", "longitude")})
        for name in merged.coords:
            merged[name].encoding = _encoding(reference[name])

        sources = []
        for path, ds in zip(files, datasets):
            for name in ("latitude", "longitude"):
                if not np.array_equal(ds[name].values, reference[name].values):
                    raise ValueError(f"{path.name} is not on the grid of the other files")
            for name, variable in ds.data_vars.items():
                if name in merged:
                    raise ValueError(f"Variable {name!r} exists in more than one file")
                attrs = dict(variable.attrs)
                if tuple(ds[TIME].values) != shared_axis:
                    attrs["reskit_original_time_start"] = str(ds[TIME].values[0])[:19]
                # positional, the way Era5Source reads the files
                merged[name] = xr.Variable(variable.dims, variable.load().values, attrs)
                merged[name].encoding = _encoding(variable)
            sources.append(f"{path.relative_to(FIXTURES).as_posix()} sha256:{_sha256(path)}")

        merged.attrs = dict(reference.attrs)
        merged.attrs["reskit_source_files"] = sources
        merged.attrs["reskit_created_by"] = "scripts/make_era5_zarr_test_data.py"
        return merged
    finally:
        for ds in datasets:
            ds.close()


def write_store(dataset: xr.Dataset, store: Path) -> None:
    """Write a dataset as a deterministic, consolidated Zarr format 3 store.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset returned by build_dataset
    store : Path
        The store to create, an existing store is replaced
    """
    if store.exists():
        shutil.rmtree(store)

    encoding = {}
    for name, variable in dataset.variables.items():
        encoding[name] = dict(variable.encoding, chunks=variable.shape, compressors=(COMPRESSOR,))
    for variable in dataset.variables.values():
        variable.encoding = {}

    # Era5ZarrSource opens stores with consolidated metadata by default. zarr warns that
    # this is not (yet) part of the format 3 specification, which is known and accepted.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
        dataset.to_zarr(store, mode="w", zarr_format=3, consolidated=True, encoding=encoding)


def make_stores(root: Path) -> list[Path]:
    """Build every store of STORES below 'root'.

    Parameters
    ----------
    root : Path
        The directory to write the stores to, mirroring the layout of FIXTURES

    Returns
    -------
    list of Path
        The stores written
    """
    written = []
    for store, (pattern, time_axis_from) in STORES.items():
        files = sorted(FIXTURES.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No netCDF4 fixtures match {FIXTURES / pattern}")
        target = root / store
        target.parent.mkdir(parents=True, exist_ok=True)
        write_store(build_dataset(files, time_axis_from), target)
        written.append(target)
    return written


def _differences(expected: Path, actual: Path) -> list[str]:
    """List the files which differ between two directory trees."""
    comparison = filecmp.dircmp(expected, actual)
    found = [f"only in the committed store: {expected / name}" for name in comparison.left_only]
    found += [f"missing from the committed store: {expected / name}" for name in comparison.right_only]
    _, mismatch, errors = filecmp.cmpfiles(expected, actual, comparison.common_files, shallow=False)
    found += [f"differs: {expected / name}" for name in mismatch + errors]
    for name in comparison.common_dirs:
        found += _differences(expected / name, actual / name)
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild the stores in a temporary directory and compare them with the committed ones, byte for byte",
    )
    args = parser.parse_args(argv)

    if not args.check:
        for store in make_stores(FIXTURES):
            print(f"wrote {store.relative_to(FIXTURES.parents[4])}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        differences = []
        for rebuilt in make_stores(Path(tmp)):
            committed = FIXTURES / rebuilt.relative_to(tmp)
            if not committed.is_dir():
                differences.append(f"missing: {committed}")
            else:
                differences += _differences(committed, rebuilt)
    for difference in differences:
        print(difference)
    print("The Zarr fixtures are up to date." if not differences else "Rerun the script to update the Zarr fixtures.")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
