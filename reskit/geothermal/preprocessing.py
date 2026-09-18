"""Build the Franzmann geothermal resource grids from their original inputs.

The temperature model is equation (1), section 2.3, of Franzmann, Heinrichs
and Stolten (2025), https://doi.org/10.1016/j.renene.2025.123199. The fixed
9 km correction to sustainable heat flow comes from Franzmann's archived
``calcualte_sustainable_heat.py``; it is not specified in the paper.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal, osr

__all__ = ["create_geothermal_resource"]

_LAT = np.arange(-89.5, 90, 1)
_LON = np.arange(-179.5, 180, 1)
_PAPER = "https://doi.org/10.1016/j.renene.2025.123199"
_CONDUCTIVITY_COLUMN = "Thermal Conductivity [W/mK] [2]"


def _read_heat_fields(path):
    """Read the original Goutorbe two-header table in geographic order."""
    table = pd.read_csv(path, comment="#", header=[0, 1])
    if list(table.columns[:2]) != [("lon", "degree"), ("lat", "degree")]:
        raise ValueError("Goutorbe table must start with lon, lat columns in degrees.")
    table = table.set_index(list(table.columns[:2]))
    expected = pd.MultiIndex.from_product([_LON, _LAT])
    if table.index.has_duplicates or len(table) != len(expected) or not expected.isin(table.index).all():
        raise ValueError("Goutorbe table must contain every global 1-degree cell exactly once.")
    arrays = []
    for column in [("Similarity method: mean HF", "mW/m2"), ("Heat prod (provinces)", "uW/m3")]:
        if column not in table:
            raise ValueError(f"Missing Goutorbe column or incorrect units: {column!r}.")
        # unstack longitude to give ascending latitude rows and longitude columns.
        field = table[column].unstack(level=0).reindex(index=_LAT, columns=_LON)
        arrays.append(field.to_numpy(dtype=np.float64))
    return arrays


def _average_to_one_degree(values):
    """Average four aligned half-degree cells using double precision."""
    return values.astype(np.float64).reshape(180, 2, 360, 2).mean(axis=(1, 3))


def _read_conductivity(lithology_path, conductivity_table):
    """Map GLiM classes using the applied first worksheet, then average."""
    mapping = pd.read_excel(conductivity_table, sheet_name=0, index_col=0).iloc[:16]
    if not np.array_equal(mapping.index, np.arange(1, 17)) or _CONDUCTIVITY_COLUMN not in mapping:
        raise ValueError("Conductivity workbook must contain GLiM classes 1 to 16 and their W/mK column.")
    coefficients = mapping[_CONDUCTIVITY_COLUMN].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(coefficients) & (coefficients > 0)):
        raise ValueError("All conductivity coefficients must be finite and positive.")

    raster = gdal.Open(str(lithology_path), gdal.GA_ReadOnly)
    if raster is None:
        raise ValueError(f"Cannot open lithology raster: {lithology_path}")
    try:
        if (
            raster.RasterCount != 1
            or (raster.RasterYSize, raster.RasterXSize) != (360, 720)
            or not np.allclose(raster.GetGeoTransform(), (-180, 0.5, 0, 90, 0, -0.5), rtol=0, atol=1e-10)
        ):
            raise ValueError("GLiM must be the aligned global 0.5-degree raster with north-to-south rows.")
        srs = raster.GetSpatialRef()
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        # The original ASCII grid has no CRS sidecar. If declared, check it.
        if srs is not None and (not srs.IsGeographic() or not srs.IsSameGeogCS(wgs84)):
            raise ValueError("GLiM coordinates must use WGS84 longitude and latitude.")
        band = raster.GetRasterBand(1)
        classes = band.ReadAsArray()
        nodata = band.GetNoDataValue()
    finally:
        raster = None

    missing = ~np.isfinite(classes)
    if nodata is not None:
        missing |= classes == nodata
    if not np.all(missing | np.isin(classes, np.arange(1, 17))):
        raise ValueError("GLiM contains an unknown lithology class.")
    # Class 15 is 'No Data'. The archived workbook uses 2.5 W/(m K) here
    # and for water bodies, whose reference value was changed from 0.6.
    conductivity = np.full(classes.shape, coefficients[14], dtype=np.float64)
    for code, value in enumerate(coefficients, start=1):
        conductivity[classes == code] = value
    return np.flip(_average_to_one_degree(conductivity), axis=0)


def _read_surface_temperature(path):
    """Read TS climatology slice 13, repair the missing edge and average."""
    with xr.open_dataset(path, decode_times=False) as source:
        if "TS" not in source or set(source.TS.dims) != {"time", "lat", "lon"}:
            raise ValueError("NASA POWER input must contain TS(time, lat, lon).")
        if source.TS.attrs.get("units") not in {"C", "degC", "degree_Celsius", "degrees_Celsius"}:
            raise ValueError("NASA POWER TS must be in degrees Celsius.")
        if 13 not in source.time.values:
            raise ValueError("NASA POWER input must contain climatology time=13.")
        source = source.sortby("lat").sortby("lon")
        if not np.array_equal(source.lat.values, np.arange(-89.75, 90, 0.5)) or not np.array_equal(
            source.lon.values, np.arange(-179.75, 180, 0.5)
        ):
            raise ValueError("NASA POWER input must use the aligned global 0.5-degree cell centres.")
        values = source.TS.sel(time=13).transpose("lat", "lon").values.astype(np.float64)
    # The archived delivery has a completely missing final longitude column.
    # Preserve valid values if a repaired delivery is supplied.
    edge_missing = np.isnan(values[:, -1])
    values[edge_missing, -1] = values[edge_missing, -2]
    return _average_to_one_degree(values)


def _source_metadata(paths):
    """Record portable filenames and hashes rather than machine-specific paths."""
    metadata = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata[name] = {"filename": path.name, "sha256": digest.hexdigest()}
    return json.dumps(metadata, sort_keys=True)


def create_geothermal_resource(
    goutorbe_table,
    lithology_raster,
    surface_temperature,
    conductivity_table,
    output_dir=None,
):
    """Create the global 1-degree resource grids used by the EGS workflow.

    Parameters
    ----------
    goutorbe_table : str or pathlib.Path
        Original ``Supplementary material.txt`` from Goutorbe et al. (2011),
        including its two header rows. Uses similarity-method mean heat flow
        in mW/m2 and province-based heat production in uW/m3.
    lithology_raster : str or pathlib.Path
        GLiM v1.0 ``glim_wgs84_0point5deg.txt.asc`` or an equivalent 0.5-degree
        WGS84 raster with unchanged class values and geographic alignment.
    surface_temperature : str or pathlib.Path
        Archived NASA POWER ``POWER_Global_Climatology_LST_2.nc``, containing
        Earth Skin Temperature (TS, Celsius) on the global 0.5-degree grid.
        The historical climatology period is not recorded in that delivery.
    conductivity_table : str or pathlib.Path
        Franzmann's ``Mapper_Lithologic_to_heat_conductivity.xlsx``. Reads
        the first worksheet, including the applied water-body override of
        2.5 W/(m K), and uses class 15's coefficient for raster nodata.
    output_dir : str or pathlib.Path, optional
        Also write ``Temperatures.nc4`` and
        ``heat_flow_sustainable_W_per_m2.nc4`` here. Existing output files
        are refused. With no directory, return the data without writing it.

    Returns
    -------
    xarray.Dataset
        ``temperature(lat, lon, depth)`` at 1000 to 10000 m in 1000 m steps,
        ``surface_temperature(lat, lon)`` in Celsius, and
        ``heat_flow_sustainable_W_per_m2(lat, lon)``. Latitude and longitude
        ascend. Units, references, input hashes and processing notes are
        included in the returned dataset and the written files.

    Notes
    -----
    Temperature follows section 2.3, equation (1), of Franzmann et al. (2025):
    ``T(z) = Tsurf + q*z/k - A*z**2/(2*k)``. Heat flow and heat production
    are converted to SI units first. GLiM conductivity and NASA temperature
    are averaged over aligned 2x2 cells, after filling missing values in the
    last NASA longitude column from its neighbour.

    The fixed ``q - 9000*A`` sustainable-flow correction is from the archived
    script, not an equation stated in the paper. The original NaN mask is
    retained even at zero depth: missing heat production also masks the
    saved surface temperature. Regeneration from the original GLiM grid
    may differ from the archived intermediate raster by floating-point
    rounding. New provenance attributes also change the NetCDF file hashes.

    References
    ----------
    Franzmann, Heinrichs and Stolten (2025), Renewable Energy 250, 123199.
    https://doi.org/10.1016/j.renene.2025.123199

    Goutorbe et al. (2011), Geophysical Journal International 187, 1405-1419.
    https://doi.org/10.1111/j.1365-246X.2011.05228.x

    Hartmann and Moosdorf (2012), GLiM v1.0, PANGAEA.
    https://doi.org/10.1594/PANGAEA.788537
    """
    paths = {
        "goutorbe_table": Path(goutorbe_table),
        "lithology_raster": Path(lithology_raster),
        "surface_temperature": Path(surface_temperature),
        "conductivity_table": Path(conductivity_table),
    }
    outputs = []
    if output_dir is not None:
        outputs = [Path(output_dir) / name for name in ("Temperatures.nc4", "heat_flow_sustainable_W_per_m2.nc4")]
        for path in outputs:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite geothermal resource: {path}")

    heat_flow, heat_production = _read_heat_fields(paths["goutorbe_table"])
    conductivity = _read_conductivity(paths["lithology_raster"], paths["conductivity_table"])
    surface = _read_surface_temperature(paths["surface_temperature"])
    profiles = []
    # Preserve the archived order of floating-point operations and NaN handling.
    for depth in range(0, 10001, 1000):
        profiles.append(
            surface + heat_flow * 1e-3 * depth / conductivity - heat_production * 1e-6 * depth**2 / 2 / conductivity
        )

    result = xr.Dataset(
        {
            "temperature": (("lat", "lon", "depth"), np.stack(profiles[1:], axis=-1)),
            "surface_temperature": (("lat", "lon"), profiles[0]),
            "heat_flow_sustainable_W_per_m2": (("lat", "lon"), heat_flow * 1e-3 - 9000 * heat_production * 1e-6),
        },
        coords={"lat": _LAT, "lon": _LON, "depth": np.arange(1000, 10001, 1000)},
        attrs={
            "title": "Global subsurface temperature and sustainable geothermal heat flow",
            "references": (
                f"{_PAPER}; https://doi.org/10.1111/j.1365-246X.2011.05228.x; "
                "https://doi.org/10.1594/PANGAEA.788537; https://power.larc.nasa.gov/"
            ),
            "source": "David Franzmann's archived dissertation preprocessing scripts (2022)",
            "source_files": _source_metadata(paths),
            "history": f"{datetime.now(timezone.utc).isoformat()}: reskit.geothermal.preprocessing.create_geothermal_resource",
            "temperature_method": "Franzmann et al. (2025), section 2.3, equation (1): T(z)=Tsurf+q*z/k-A*z^2/(2*k)",
            "sustainable_heat_flow_method": (
                "q-9000*A in SI units; archived calcualte_sustainable_heat.py. "
                "The 9000 m correction is not specified in the paper."
            ),
            "preprocessing": (
                "GLiM classes mapped using the workbook's first worksheet; raster nodata uses class 15. "
                "NASA TS time=13 selected; missing last longitude column filled from its neighbour. "
                "Both 0.5-degree grids averaged to 1 degree. Original heat-production NaN mask retained."
            ),
            "nasa_climatology_period": "Not recorded in the archived input; do not infer from current POWER releases.",
        },
    )
    result.lat.attrs = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    result.lon.attrs = {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    result.depth.attrs = {"long_name": "Depth below surface", "units": "m", "positive": "down", "axis": "Z"}
    result.temperature.attrs = {"long_name": "Rock temperature at depth", "units": "degree_Celsius"}
    result.surface_temperature.attrs = {
        "long_name": "Mean surface temperature with resource mask",
        "units": "degree_Celsius",
    }
    result.heat_flow_sustainable_W_per_m2.attrs = {
        "long_name": "Sustainable geothermal heat flow using the archived 9 km correction",
        "units": "W m-2",
    }
    if outputs:
        outputs[0].parent.mkdir(parents=True, exist_ok=True)
        result[["temperature", "surface_temperature"]].to_netcdf(outputs[0])
        result[["heat_flow_sustainable_W_per_m2"]].to_netcdf(outputs[1])
    return result
