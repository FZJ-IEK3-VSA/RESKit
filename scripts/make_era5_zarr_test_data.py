"""Cut the ERA5 Zarr test store from an online ERA5 Zarr store.

The stores of scripts/make_era5_zarr_test_data.py are converted from the netCDF4
fixtures, so they share the netCDF4 layout: a 'time' axis, longitudes on
[-180, 180), packed integers and the radiation RESKit preprocessed. They test that
Era5ZarrSource reads the same data as Era5Source, but none of what is specific to
a real online store. This script cuts such a store, so that the tests can read real
online data the way users do. The source is SOURCE_URL, currently the ERA5
single-levels store of the Earth Data Hub (EDH):

    SOURCE_URL  ->  era5-zarr/era5.zarr

* The same box as the 'era5' fixtures (49-52 N, 5-7.5 E) and their 140 hours
  (2015-01-01 00:00 to 2015-01-06 19:00), plus the hour before, so that the tests
  select those hours with 'time_slice', as users of a long store do.
* The raw variables RESKit reads (Era5Source.CDS_TO_NC_NAME), and nothing derived.
* The layout of the source: the 'valid_time' axis, longitudes on its [0, 360) grid
  (the box lies below 180 E, so Era5ZarrSource does not need to wrap them), latitudes
  descending, float32, the variable attributes and the time encoding.
* The values bit for bit. The source applies bitround (13 mantissa bits) as a codec;
  the values it returns are already rounded, so the store drops that codec and stays
  spec-only Zarr format 3, with consolidated metadata.

The cut lies in a single source chunk per variable, so a run costs about 15 chunk
requests against the EDH quota. The source can change -- ERA5 is extended and
occasionally corrected, and the store is re-ingested -- so, unlike the converted
stores, this one cannot be rebuilt byte for byte and is meant to be cut once. The
source URL and the retrieval date are written into its attributes.

Requires EDH credentials in ~/.netrc:

    machine data.earthdatahub.destine.eu
        login <your-username>
        password <your-token>

Usage (from the repository root):

    python scripts/make_era5_zarr_test_data.py
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

SOURCE_URL = "https://data.earthdatahub.destine.eu/era5/era5-single-levels-atmosphere-v0.zarr"

FIXTURES = Path(__file__).resolve().parents[1] / "reskit" / "data" / "test_cache" / "data" / "reskit-test-data"
STORE = FIXTURES / "era5-zarr" / "era5.zarr"

TIME = "valid_time"
VARIABLES = sorted(Era5Source.CDS_TO_NC_NAME.values())
LATITUDES = (52.0, 49.0)  # descending, as in the source
LONGITUDES = (5.0, 7.5)
TIMES = ("2014-12-31 23:00", "2015-01-06 19:00")
SHAPE = (141, 13, 11)

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
    ds = xr.open_dataset(
        source,
        engine="zarr",
        chunks=None,
        storage_options={"client_kwargs": {"trust_env": True}},
        decode_timedelta=False,
    )
    try:
        selection = ds[VARIABLES].sel(
            {TIME: slice(*TIMES), "latitude": slice(*LATITUDES), "longitude": slice(*LONGITUDES)}
        )
        if tuple(selection.sizes[name] for name in (TIME, "latitude", "longitude")) != SHAPE:
            raise ValueError(f"The cut has the shape {dict(selection.sizes)}, expected {SHAPE}")
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
    encoding = {}
    for name, variable in dataset.variables.items():
        kept = {key: variable.encoding[key] for key in ("dtype", "units", "calendar") if key in variable.encoding}
        encoding[name] = dict(kept, chunks=variable.shape, compressors=(COMPRESSOR,))
    for variable in dataset.variables.values():
        variable.encoding = {}

    dataset.attrs.update(
        reskit_source_url=source,
        reskit_retrieved=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        reskit_selection=(
            f"{TIME} {TIMES[0]} to {TIMES[1]}, latitude {LATITUDES[0]} to {LATITUDES[1]}, "
            f"longitude {LONGITUDES[0]} to {LONGITUDES[1]}"
        ),
        reskit_created_by="scripts/make_era5_zarr_test_data.py",
    )

    if store.exists():
        shutil.rmtree(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    # Era5ZarrSource opens stores with consolidated metadata by default. zarr warns that
    # this is not (yet) part of the format 3 specification, which is known and accepted.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
        dataset.to_zarr(store, mode="w", zarr_format=3, consolidated=True, encoding=encoding)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, default=STORE, help=f"The store to write, default {STORE}")
    args = parser.parse_args(argv)

    write_store(cut(), args.store)
    print(f"wrote {args.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
