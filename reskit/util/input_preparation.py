import reskit as rk

depends_on = {
    "wind_era5_PenaSanchezDunkelWinklerEtAl2025":
        {
            "GWA4": ["50m","100m","200m"],
            "ERA5": ["100m_u_component_of_wind", "100m_v_component_of_wind", "2m_temperature", "surface_pressure", "boundary_layer_height"]
        },
    "onshore_wind_iconlam_2023": ["ICONLAM"],
    "onshore_wind_merra_ryberg2019_europe": ["MERRA-2","GWA2"],
}
 
def download_and_process(
        workflow, 
        start_date, 
        end_date, 
        boundary_box, 
        output_dir, 
        tiling=False, 
        zoom_level=4):
   

    output_paths = {
        "rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025":
            {"era5_path": rk.preparing_era5(
                                        start_date=start_date,
                                        end_date=end_date,
                                        boundary_box=boundary_box,
                                        output_dir=output_dir,
                                        variables=depends_on[workflow]["ERA5"],
                                    ),
             "height_scaling_data": {
                 50: "some_path_50m",
                100: "some_path_100m",
                200: "some_path_200m"
             }
            },
    }
    
    return output_paths[workflow]
