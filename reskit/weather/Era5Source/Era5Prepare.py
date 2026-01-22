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
2. The CDO (Climate Data Operators) installed on your system. Install python-cdo from channel conda-forge for example: conda install -c conda-forge python-cdo
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


def preprocess_era5_data(era5_dir: str):
    # prepare cdo instance
    cdo = Cdo()

    # find nc files
    nc_files = sorted(f for f in os.listdir(era5_dir) if f.endswith(".nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {era5_dir}")
    # iterate nc files
    for fname in nc_files:
        input_nc = os.path.join(era5_dir, fname)
        # detect variables
        varnames = cdo.showname(input=input_nc)[0].split()
        varset = set(varnames)

        # process for solar radiation variables
        """
        ssrd and fdir are hourly backward accumulated quantities in ERA5 with the unit: J m⁻². 
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
        if "ssrd" in varset:
            unit = "W m**-2"
            cdo.divc(
                3600,
                input=f"-selname,ssrd {input_nc}",
                options=f"-L -setattribute,ssrd@units={unit}",
                output=os.path.join(
                    era5_dir,
                    f"{fname.split('.')[0]}_processed_ssrd.nc",
                ),
            )
            cdo.chname(
                "ssrd,ssrd_t_adj",
                input=f"-selname,ssrd -shifttime,+1hour -divc,3600 -setattribute,ssrd@units={unit} {input_nc}",
                options="-L",
                output=os.path.join(
                    era5_dir,
                    f"{fname.split('.')[0]}_processed_ssrd_t_adjusted.nc",
                ),
            )
        if "fdir" in varset:
            unit = "W m**-2"
            cdo.divc(
                3600,
                input=f"-selname,fdir {input_nc}",
                options=f"-L -setattribute,fdir@units={unit}",
                output=os.path.join(
                    era5_dir,
                    f"{fname.split('.')[0]}_processed_fdir.nc",
                ),
            )
            cdo.chname(
                "fdir,fdir_t_adj",
                input=f"-selname,fdir -shifttime,+1hour -divc,3600 -setattribute,fdir@units={unit} {input_nc}",
                options="-L",
                output=os.path.join(
                    era5_dir,
                    f"{fname.split('.')[0]}_processed_fdir_t_adjusted.nc",
                ),
            )

        # process for wind speed variables
        if {"u100", "v100"} <= varset:
            cdo.expr(
                "'ws100=sqrt(u100*u100+v100*v100)'",
                input=input_nc,
                output=os.path.join(
                    era5_dir,
                    f"{fname.split('.')[0]}_processed_ws100.nc",
                ),
            )
        if {"u10", "v10"} <= varset:
            cdo.expr(
                "'ws10=sqrt(u10*u10+v10*v10)'",
                input=input_nc,
                output=os.path.join(
                    era5_dir,
                    f"{fname.split('.')[0]}_processed_ws10.nc",
                ),
            )


def preparing_era5(
    start_date: str,
    end_date: str,
    boundary_box: dict,
    output_dir: str,
    variables: Optional[List[str]] = None,
):
    # 1. download ERA5 data for the given date range and boundary box
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

    output_file = os.path.join(
        output_dir, f"{era5_dataset}_{dates[0].strftime('%Y%m')}-{dates[-1].strftime('%Y%m')}.nc"
    )
    if not os.path.exists(output_file):
        era5_downloader(
            target_filename=output_file,
            year=years,
            month=months,
            variables=variables,
            area=area,
        )

        # 2. preprocess ERA5 data
        preprocess_era5_data(era5_dir=output_dir)
    else:
        print(f"ERA5 data already exists at {output_file}, skipping download and preprocessing.")

    return output_dir
