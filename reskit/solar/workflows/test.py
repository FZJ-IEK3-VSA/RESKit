import reskit as rk
import pandas as pd
import time

WinklerUnpublished = True
newPOA = True # in Winkler unpublished, a new POA calc method is available via solarfactors
new_branch = True

singleaxis = False

year=2012


# Make a placements dataframe
placements = pd.DataFrame()
placements["lon"] = [
    # 8.649, 
    #6.083,
    6.481105, #Düren
]  # Longitude
placements["lat"] = [
    # 50.270, 
    #50.775,
    50.80295 # Düren
]  # Latitude
placements["modtilt"] = [
    # 32,
    36, # optimum tilt düren laut GSA
]  # System tilt in degrees
placements["modazimuth"] = [
    # 180,
    157, # module azimuth real plant Düren
]  # System azimuth in degrees
placements["capacity"] = [
    2000,
]  # Total system capacity in kW
# placements["elev"] = [
#     # 250,
#     125, #düren
# ]  # Altitute in meters

if not new_branch:
    # new wf did not yet exist in old branch
    assert not WinklerUnpublished
    assert not newPOA
    # old args were tilt and azimuth only
    placements = placements.rename(columns={"modtilt":"tilt", "modazimuth":"azimuth"})

workflow_args = {
    "placements" : placements,
    "era5_path" : f"/benchtop/shared_data/weather_data/processed_weather_data/ERA5_global_processed_V2022.02/4/<X-TILE>/<Y-TILE>/{str(year)}/reanalysis-era5-single-levels.z4.x<X-TILE>.y<Y-TILE>.y{str(year)}.*.nc",
    "global_solar_atlas_ghi_path" : "/benchtop/shared_data/2023_gears/geography/irradiance/global_solar_atlas_v2.9/World_GHI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/GHI.tif",
    "global_solar_atlas_dni_path" : "/benchtop/shared_data/2023_gears/geography/irradiance/global_solar_atlas_v2.9/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif",
    "module" : "LG Electronics LG370Q1C-A5", 
    # "module" : "WINAICO WSx-240P6",
    "elev" : 300,
    "inverter" : None,
    "inverter_kwargs" : {},
    "DNI_nodata_fallback" : 1.0,
    "DNI_nodata_fallback_scaling" : 1.0,
    "GHI_nodata_fallback" : 1.0,
    "GHI_nodata_fallback_scaling" : 1.0,
    "output_netcdf_path" : None,
    "output_variables": None,
    "tech_year" : 2019
    }

if new_branch:
    # we have the latest branch with 2 different workflows and 2 POA options in Winkler unpublished
    if WinklerUnpublished:
        workflow_args["ground_albedo"] = 0.25 #TODO ("esa_cci_v2.1.1", "/benchtop/shared_data/2023_gears/geography/landcover/esa_cci_v2.1.1/C3S-LC-L4-LCCS-Map-300m-P1Y-2018-v2.1.1.tif")
        workflow_args["bifaciality_factor"] = None #0.9
        if newPOA:
            workflow_args["new_style"] = True
        else:
            workflow_args["new_style"] = False
    elif newPOA:
        # old wf + new POA is not the actual historic solution, warn!
        print("##################################################################################################")
        print("WARNING: openfield_pv_era5() workflow did originally not have the new POA option via solarfactors!")
        print("##################################################################################################")


if singleaxis:
    workflow_args["tracking"] = "single_axis"
    workflow_args["tracking_args"] = {"max_angle": 60}
else:
    workflow_args["tracking"] = "fixed"


if WinklerUnpublished:
    wf = rk.solar.pv_era5_WinklerUnpublished
else:
    wf = rk.solar.openfield_pv_era5

_start = time.time()
out = rk.execute_workflow_iteratively(
    workflow=wf, weather_path_varname="era5_path", zoom=4, **workflow_args
    )
_end = time.time()
print(f"{'New' if WinklerUnpublished else 'Old'} workflow execution on {'new' if new_branch else 'old'} branch with {'new' if newPOA else 'old'} POA method took {_end-_start} seconds and yields {out['capacity_factor'].values.sum()} FLH ({'Bifaciality factor : '+str(workflow_args['bifaciality_factor'])}).")
pass