"""Create the ERA5 Zarr test data subset (era5-zarr/era5.zarr) from an online ERA5 store.

This script was used to cut the subset once from the ERA5 single-levels Zarr store of
the Earth Data Hub (https://data.earthdatahub.destine.eu/era5/era5-single-levels-atmosphere-v0.zarr). 
The subset holds the raw ERA5 variables RESKit reads (Era5Source.CDS_TO_NC_NAME) for the 
box of the 'era5' fixtures (49-52 N, 5-7.5 E) and their 140 hours 
(2015-01-01 00:00 to 2015-01-06 19:00), plus the hour before. It keeps
the layout and values of the source ('valid_time' axis, longitudes on [0, 360),
descending latitudes, float32), so that the tests read real online data the way users
do. The source URL and the retrieval date are stored in its attributes.

Run it from the repository root, with EDH credentials in ~/.netrc
(machine data.earthdatahub.destine.eu, login <username>, password <token>):

    python scripts/make_era5_zarr_test_data.py

The store is the bundle member reskit-test-data/era5-zarr, so record a new cut with

    reskit-data bundle update reskit/data/test_cache
"""

import argparse
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec

from reskit.weather import Era5Source

# Online source store and the test store inside the bundled test data
SOURCE_URL = "https://data.earthdatahub.destine.eu/era5/era5-single-levels-atmosphere-v0.zarr"

FIXTURES = Path(__file__).resolve().parents[1] / "reskit" / "data" / "test_cache" / "data" / "reskit-test-data"
STORE = FIXTURES / "era5-zarr" / "era5.zarr"

# The subset: the raw variables RESKit reads, in the box and hours of the 'era5' fixtures
TIME = "valid_time"
VARIABLES = sorted(Era5Source.CDS_TO_NC_NAME.values())
LATITUDES = (52.0, 49.0)  # descending, as in the source
LONGITUDES = (5.0, 7.5)
TIMES = ("2014-12-31 23:00", "2015-01-06 19:00")
SHAPE = (141, 13, 11)  # expected (time, latitude, longitude) sizes of the cut

# Compression of the written store
COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")


def cut(source: str = SOURCE_URL) -> xr.Dataset:
    """Read the test box from the source store.

    Parameters
    ----------
    source : str, optional
        The store to cut from

    Returns
    -------
    xarray.Dataset
        The loaded cut, one data variable per raw ERA5 variable RESKit reads
    """
    # Open the online store lazily, without dask and without decoding timedeltas
    ds = xr.open_dataset(
        source,
        engine="zarr",
        chunks=None,
        storage_options={"client_kwargs": {"trust_env": True}},
        decode_timedelta=False,
    )
    try:
        # Select the variables, hours and box of the subset
        box = {
            TIME: slice(*TIMES),
            "latitude": slice(*LATITUDES),
            "longitude": slice(*LONGITUDES),
        }
        selection = ds[VARIABLES].sel(box)

        # Check the selection before any data is downloaded
        shape = tuple(selection.sizes[name] for name in box)
        if shape != SHAPE:
            raise ValueError(f"The cut has the shape {dict(selection.sizes)}, expected {SHAPE}")

        # Download the selected chunks
        selection = selection.load()
    finally:
        ds.close()

    # zarr fills a chunk the server does not deliver with NaN, without an error
    for name, variable in selection.data_vars.items():
        if np.isnan(variable.values).any():
            raise ValueError(f"{name!r} contains NaN, a source chunk may not have been delivered")
    return selection


def write_store(dataset: xr.Dataset, store: Path, source: str = SOURCE_URL) -> None:
    """Write the cut as a consolidated Zarr format 3 store, one chunk per variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset returned by cut
    store : Path
        The store to create, an existing store is replaced
    source : str, optional
        The store the cut was read from, recorded in the attributes
    """
    # Keep the dtype and time encoding of the source, write one compressed chunk per variable
    encoding = {}
    for name, variable in dataset.variables.items():
        kept = {}
        for key in ("dtype", "units", "calendar"):
            if key in variable.encoding:
                kept[key] = variable.encoding[key]
        encoding[name] = dict(kept, chunks=variable.shape, compressors=(COMPRESSOR,))
    # Drop the remaining source encoding, such as its bitround codec, as the values are already rounded
    for variable in dataset.variables.values():
        variable.encoding = {}

    # Record where, when and how the subset was cut
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    selection = (
        f"{TIME} {TIMES[0]} to {TIMES[1]}, "
        f"latitude {LATITUDES[0]} to {LATITUDES[1]}, "
        f"longitude {LONGITUDES[0]} to {LONGITUDES[1]}"
    )
    dataset.attrs.update(
        reskit_source_url=source,
        reskit_retrieved=retrieved,
        reskit_selection=selection,
        reskit_created_by="scripts/make_era5_zarr_test_data.py",
    )

    # Replace an existing store
    if store.exists():
        shutil.rmtree(store)
    store.parent.mkdir(parents=True, exist_ok=True)

    # Write the store. Era5ZarrSource opens stores with consolidated metadata by default. zarr warns that
    # this is not (yet) part of the format 3 specification, which is known and accepted.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
        dataset.to_zarr(
            store,
            mode="w",
            zarr_format=3,
            consolidated=True,
            encoding=encoding,
        )


def main(argv=None) -> int:
    # Command line: the store to write can be changed
    description = __doc__.split("\n\n")[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--store",
        type=Path,
        default=STORE,
        help=f"The store to write, default {STORE}",
    )
    args = parser.parse_args(argv)

    # Cut the subset from the online store and write it
    dataset = cut()
    write_store(dataset, args.store)
    print(f"wrote {args.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
