import os
import cdsapi
from cdo import Cdo
import pandas as pd
from typing import Union, List, Optional

"""
Running this module will automatically download ERA5 data from CDS for the specified date range and boundary box,
and preprocess the data to adjust solar radiation variables and compute wind speed variables.

To make it happen, you need to have:
1. An account at the Copernicus Climate Data Store (CDS) and have your API key set up in the ~/.cdsapirc file. See here for more details: https://cds.climate.copernicus.eu/how-to-api
2. The CDO (Climate Data Operators) installed on your system. Install cdo and python-cdo from channel conda-forge will do the job, for example: conda install -c conda-forge cdo python-cdo
"""


era5_variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
    "boundary_layer_height",
    "forecast_surface_roughness",
]
era5_dataset = "reanalysis-era5-single-levels"


def era5_downloader(
    target_filename: str,
    year: Union[str, List[str]],
    month: Union[str, List[str]],
    variables: list[str],
    area: tuple[float, float, float, float],
    grid: tuple[float, float] = (0.25, 0.25),
    data_format: str = "netcdf_legacy",
):
    client = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "format": data_format,
        "variable": variables,
        "year": year,
        "month": month,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,  # [north, west, south, east]
        "grid": f"{grid[0]}/{grid[1]}",
    }

    client.retrieve(
        era5_dataset,
        request,
        target_filename,
    )


def preprocess_era5_data(focus_nc: str):
    """
    Ssrd and fdir are hourly backward accumulated quantities in ERA5 with the unit: J m⁻².
    Each value at time t represents the accumulated energy over the previous hour.

    When you divide it by 3600 seconds, you convert the accumulated energy (J m⁻²)
    into an average power flux (W m⁻²) over that hour.

    However, in solar observation and other models,
    this value represents the instantaneous mean over the next hour.
    This matches:
        PV modeling conventions
        Many energy system models
        atlite / PyPSA conventions

    So, we need to do two things:
    1. Convert the accumulated quantity to an average power flux by dividing by 3600
    2. Shift the time axis forward by one hour to represent the average over the next hour.
    """
    # prepare cdo instance
    cdo = Cdo()

    # helper function to check variables in nc file, to skip processing if already done
    def nc_file_has_vars(cdo, nc_path, required_vars):
        """Check whether a NetCDF file exists and contains required variables."""
        if not os.path.exists(nc_path):
            return False
        try:
            vars_in_file = set(cdo.showname(input=nc_path)[0].split())
            return set(required_vars) <= vars_in_file
        except Exception:
            return False

    # detect variables in the nc file
    varnames = cdo.showname(input=focus_nc)[0].split()
    varset = set(varnames)

    dir = os.path.dirname(focus_nc)
    f_name = os.path.basename(focus_nc)
    # process for solar radiation variables
    # solar_out = os.path.join(dir, f"{f_name.split('.')[0]}_processed_solar.nc")
    # if {"ssrd", "fdir"} & varset:
    #     if not nc_file_has_vars(cdo, solar_out, ["ssrd", "fdir"]):
    #         unit = "W m**-2"
    #         cdo.copy(
    #             input=(
    #                 f"-setattribute,ssrd@units=\"{unit}\" "
    #                 f"-setattribute,fdir@units=\"{unit}\" "
    #                 f"-divc,3600 "
    #                 f"-selname,ssrd,fdir "
    #                 f"{focus_nc}"
    #             ),
    #             output=solar_out,
    #         )
    #     else:
    #         print(f"Skipping solar preprocessing (exists): {solar_out}")

    # process for solar radiation variables (time adjusted)
    solar_t_out = os.path.join(dir, f"{f_name.split('.')[0]}_processed_solar_t_adjusted.nc")
    if {"ssrd", "fdir"} & varset:
        if not nc_file_has_vars(cdo, solar_t_out, ["ssrd_t_adj", "fdir_t_adj"]):
            unit = "W m**-2"
            cdo.copy(
                input=(
                    f"-chname,ssrd,ssrd_t_adj,fdir,fdir_t_adj "
                    f'-setattribute,ssrd@units="{unit}" '
                    f'-setattribute,fdir@units="{unit}" '
                    f"-shifttime,+1hour "
                    f"-divc,3600 "
                    f"-selname,ssrd,fdir "
                    f"{focus_nc}"
                ),
                output=solar_t_out,
            )
        else:
            print(f"Skipping process time-adjusted solar (exists): {solar_t_out}")

    # process for wind speed variables
    ws100_out = os.path.join(dir, f"{f_name.split('.')[0]}_processed_ws100.nc")
    if {"u100", "v100"} <= varset:
        if not nc_file_has_vars(cdo, ws100_out, ["ws100"]):
            unit = "m s**-1"
            long_name = "100 metre wind speed"
            cdo.copy(
                input=(
                    f'-setattribute,ws100@long_name="{long_name}" '
                    f'-setattribute,ws100@units="{unit}" '
                    f"-expr,'ws100=sqrt(u100*u100+v100*v100)' "
                    f"{focus_nc}"
                ),
                output=ws100_out,
            )
        else:
            print(f"Skipping process ws100 (exists): {ws100_out}")

    ws10_out = os.path.join(dir, f"{f_name.split('.')[0]}_processed_ws10.nc")
    if {"u10", "v10"} <= varset:
        if not nc_file_has_vars(cdo, ws10_out, ["ws10"]):
            unit = "m s**-1"
            long_name = "10 metre wind speed"
            cdo.copy(
                input=(
                    f'-setattribute,ws10@long_name="{long_name}" '
                    f'-setattribute,ws10@units="{unit}" '
                    f"-expr,'ws10=sqrt(u10*u10+v10*v10)' "
                    f"{focus_nc}"
                ),
                output=ws10_out,
            )
        else:
            print(f"Skipping process ws10 (exists): {ws10_out}")


def prepare_era5(
    start_date: str,
    end_date: str,
    boundary_box: dict,
    output_dir: str,
    variables: Optional[List[str]] = None,
):
    # 1. download ERA5 data for the given date range and boundary box
    # initial preparation
    os.makedirs(output_dir, exist_ok=True)

    # resolve variables
    if variables is None:
        variables = era5_variables

    dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    years = sorted({str(d.year) for d in dates})
    months = sorted({f"{d.month:02d}" for d in dates})

    area = (
        boundary_box["north"],
        boundary_box["west"],
        boundary_box["south"],
        boundary_box["east"],
    )

    # start download
    output_file = os.path.join(
        output_dir, f"{era5_dataset}_{dates[0].strftime('%Y%m')}-{dates[-1].strftime('%Y%m')}.nc"
    )
    # check if file exists
    if not os.path.exists(output_file):
        era5_downloader(
            target_filename=output_file,
            year=years,
            month=months,
            variables=variables,
            area=area,
        )
    else:
        print(f"ERA5 data already exists at {output_file}, skipping download.")

    # 2. preprocess ERA5 data
    preprocess_era5_data(focus_nc=output_file)
    print("era5 preparation done.")

    return output_dir
