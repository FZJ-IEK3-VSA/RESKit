from reskit.solar.workflows.workflows import (
    openfield_pv_era5,
    openfield_pv_merra_ryberg2019,
    openfield_pv_sarah_unvalidated,
)
from reskit import TEST_DATA
import pytest
import numpy as np
import geokit as gk
import pandas as pd


@pytest.fixture
def pt_pv_placements() -> pd.DataFrame:
    df = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])
    df["capacity"] = 2000
    return df


def test_openfield_pv_era5(pt_pv_placements):
    gen = openfield_pv_era5(
        placements=pt_pv_placements,
        era5_path=TEST_DATA["era5-like"],
        global_solar_atlas_ghi_path=TEST_DATA["gsa-ghi-like.tif"],
        global_solar_atlas_dni_path=TEST_DATA["gsa-dni-like.tif"],
        module="WINAICO WSx-240P6",
        elev=300,
        tracking="fixed",
        inverter=None,
        inverter_kwargs={},
        tracking_args={},
        output_netcdf_path=None,
        output_variables=None,
    )

    assert gen["location"].shape == (560,)
    assert gen["capacity"].shape == (560,)
    assert gen["lon"].shape == (560,)
    assert gen["lat"].shape == (560,)
    assert gen["tilt"].shape == (560,)
    assert gen["azimuth"].shape == (560,)
    assert gen["elev"].shape == (560,)
    assert gen["time"].shape == (140,)
    assert gen["global_horizontal_irradiance"].shape == (140, 560)
    assert gen["direct_horizontal_irradiance"].shape == (140, 560)
    assert gen["surface_wind_speed"].shape == (140, 560)
    assert gen["surface_pressure"].shape == (140, 560)
    assert gen["surface_air_temperature"].shape == (140, 560)
    assert gen["surface_dew_temperature"].shape == (140, 560)
    assert gen["solar_azimuth"].shape == (140, 560)
    assert gen["apparent_solar_zenith"].shape == (140, 560)
    assert gen["direct_normal_irradiance"].shape == (140, 560)
    assert gen["extra_terrestrial_irradiance"].shape == (140, 560)
    assert gen["air_mass"].shape == (140, 560)
    assert gen["diffuse_horizontal_irradiance"].shape == (140, 560)
    assert gen["angle_of_incidence"].shape == (140, 560)
    assert gen["poa_global"].shape == (140, 560)
    assert gen["poa_direct"].shape == (140, 560)
    assert gen["poa_diffuse"].shape == (140, 560)
    assert gen["poa_sky_diffuse"].shape == (140, 560)
    assert gen["poa_ground_diffuse"].shape == (140, 560)
    assert gen["cell_temperature"].shape == (140, 560)
    assert gen["module_dc_power_at_mpp"].shape == (140, 560)
    assert gen["module_dc_voltage_at_mpp"].shape == (140, 560)
    assert gen["capacity_factor"].shape == (140, 560)
    assert gen["total_system_generation"].shape == (140, 560)

    assert np.isclose(float(gen["location"].fillna(0).mean()), 279.5)
    assert np.isclose(float(gen["capacity"].fillna(0).mean()), 2000.0)
    assert np.isclose(float(gen["lon"].fillna(0).mean()), 6.16945196229404)
    assert np.isclose(float(gen["lat"].fillna(0).mean()), 50.80320853112445)
    assert np.isclose(float(gen["tilt"].fillna(0).mean()), 39.19976325987092)
    assert np.isclose(float(gen["azimuth"].fillna(0).mean()), 180.0)
    assert np.isclose(float(gen["elev"].fillna(0).mean()), 300.0)
    assert np.isclose(
        float(gen["global_horizontal_irradiance"].fillna(0).mean()), 32.90016155215698
    )
    assert np.isclose(
        float(gen["direct_horizontal_irradiance"].fillna(0).mean()), 15.501608137870793
    )
    assert np.isclose(
        float(gen["surface_wind_speed"].fillna(0).mean()), 1.6521243123091525
    )
    assert np.isclose(
        float(gen["surface_pressure"].fillna(0).mean()), 38644.083559948376
    )
    assert np.isclose(
        float(gen["surface_air_temperature"].fillna(0).mean()), 1.0433187770747245
    )
    assert np.isclose(
        float(gen["surface_dew_temperature"].fillna(0).mean()), 0.014244860844314216
    )
    assert np.isclose(float(gen["solar_azimuth"].fillna(0).mean()), 68.6008947378997)
    assert np.isclose(
        float(gen["apparent_solar_zenith"].fillna(0).mean()), 31.143327387439044
    )
    assert np.isclose(
        float(gen["direct_normal_irradiance"].fillna(0).mean()), 51.12686118675332
    )
    assert np.isclose(
        float(gen["extra_terrestrial_irradiance"].fillna(0).mean()), 546.9559617849145
    )
    assert np.isclose(float(gen["air_mass"].fillna(0).mean()), 3.9097505883982726)
    assert np.isclose(
        float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()), 20.956699894906208
    )
    assert np.isclose(
        float(gen["angle_of_incidence"].fillna(0).mean()), 19.148843582996513
    )
    assert np.isclose(float(gen["poa_global"].fillna(0).mean()), 67.12122204692297)
    assert np.isclose(float(gen["poa_direct"].fillna(0).mean()), 37.85854851365722)
    assert np.isclose(float(gen["poa_diffuse"].fillna(0).mean()), 29.262673533265747)
    assert np.isclose(
        float(gen["poa_sky_diffuse"].fillna(0).mean()), 28.486171553638698
    )
    assert np.isclose(
        float(gen["poa_ground_diffuse"].fillna(0).mean()), 0.7765019796270453
    )
    assert np.isclose(
        float(gen["cell_temperature"].fillna(0).mean()), 2.9012114282159733
    )
    assert np.isclose(
        float(gen["module_dc_power_at_mpp"].fillna(0).mean()), 21.83819349184495
    )
    assert np.isclose(
        float(gen["module_dc_voltage_at_mpp"].fillna(0).mean()), 14.349977855519542
    )
    assert np.isclose(
        float(gen["capacity_factor"].fillna(0).mean()), 0.08039702336301864
    )
    assert np.isclose(
        float(gen["total_system_generation"].fillna(0).mean()), 160.79404672603732
    )


def test_openfield_pv_merra_ryberg2019(pt_pv_placements):
    gen = openfield_pv_merra_ryberg2019(
        placements=pt_pv_placements,
        merra_path=TEST_DATA["merra-like"],
        global_solar_atlas_ghi_path=TEST_DATA["gsa-ghi-like.tif"],
        module="WINAICO WSx-240P6",
        elev=300,
        tracking="fixed",
        inverter=None,
        inverter_kwargs={},
        tracking_args={},
        output_netcdf_path=None,
        output_variables=None,
    )

    assert gen["location"].shape == (560,)
    assert gen["capacity"].shape == (560,)
    assert gen["lon"].shape == (560,)
    assert gen["lat"].shape == (560,)
    assert gen["tilt"].shape == (560,)
    assert gen["azimuth"].shape == (560,)
    assert gen["elev"].shape == (560,)
    assert gen["time"].shape == (71,)
    assert gen["surface_wind_speed"].shape == (71, 560)
    assert gen["surface_pressure"].shape == (71, 560)
    assert gen["surface_air_temperature"].shape == (71, 560)
    assert gen["surface_dew_temperature"].shape == (71, 560)
    assert gen["global_horizontal_irradiance"].shape == (71, 560)
    assert gen["solar_azimuth"].shape == (71, 560)
    assert gen["apparent_solar_zenith"].shape == (71, 560)
    assert gen["extra_terrestrial_irradiance"].shape == (71, 560)
    assert gen["air_mass"].shape == (71, 560)
    assert gen["direct_normal_irradiance"].shape == (71, 560)
    assert gen["diffuse_horizontal_irradiance"].shape == (71, 560)
    assert gen["angle_of_incidence"].shape == (71, 560)
    assert gen["poa_global"].shape == (71, 560)
    assert gen["poa_direct"].shape == (71, 560)
    assert gen["poa_diffuse"].shape == (71, 560)
    assert gen["poa_sky_diffuse"].shape == (71, 560)
    assert gen["poa_ground_diffuse"].shape == (71, 560)
    assert gen["cell_temperature"].shape == (71, 560)
    assert gen["module_dc_power_at_mpp"].shape == (71, 560)
    assert gen["module_dc_voltage_at_mpp"].shape == (71, 560)
    assert gen["capacity_factor"].shape == (71, 560)
    assert gen["total_system_generation"].shape == (71, 560)

    print(float(gen["location"].fillna(0).mean()))
    assert np.isclose(float(gen["location"].fillna(0).mean()), 279.5)
    print(float(gen["capacity"].fillna(0).mean()))
    assert np.isclose(float(gen["capacity"].fillna(0).mean()), 2000.0)
    print(float(gen["lon"].fillna(0).mean()))
    assert np.isclose(float(gen["lon"].fillna(0).mean()), 6.16945196229404)
    print(float(gen["lat"].fillna(0).mean()))
    assert np.isclose(float(gen["lat"].fillna(0).mean()), 50.80320853112445)
    print(float(gen["tilt"].fillna(0).mean()))
    assert np.isclose(float(gen["tilt"].fillna(0).mean()), 39.19976325987092)
    print(float(gen["azimuth"].fillna(0).mean()))
    assert np.isclose(float(gen["azimuth"].fillna(0).mean()), 180.0)
    print(float(gen["elev"].fillna(0).mean()))
    assert np.isclose(float(gen["elev"].fillna(0).mean()), 300.0)
    print(float(gen["surface_wind_speed"].fillna(0).mean()))
    assert np.isclose(
        float(gen["surface_wind_speed"].fillna(0).mean()), 1.5502203948117972
    )
    print(float(gen["surface_pressure"].fillna(0).mean()))
    assert np.isclose(
        float(gen["surface_pressure"].fillna(0).mean()), 38110.883667100796
    )
    print(float(gen["surface_air_temperature"].fillna(0).mean()))
    assert np.isclose(
        float(gen["surface_air_temperature"].fillna(0).mean()), 0.6923904404714382
    )
    print(float(gen["surface_dew_temperature"].fillna(0).mean()))
    assert np.isclose(
        float(gen["surface_dew_temperature"].fillna(0).mean()), 0.2735079282721086
    )
    print(float(gen["global_horizontal_irradiance"].fillna(0).mean()))
    assert np.isclose(
        float(gen["global_horizontal_irradiance"].fillna(0).mean()), 24.425654064650278
    )
    print(float(gen["solar_azimuth"].fillna(0).mean()))
    assert np.isclose(float(gen["solar_azimuth"].fillna(0).mean()), 67.69226199649943)
    print(float(gen["apparent_solar_zenith"].fillna(0).mean()))
    assert np.isclose(
        float(gen["apparent_solar_zenith"].fillna(0).mean()), 30.755085032338677
    )
    print(float(gen["extra_terrestrial_irradiance"].fillna(0).mean()))
    assert np.isclose(
        float(gen["extra_terrestrial_irradiance"].fillna(0).mean()), 539.2545578051567
    )
    print(float(gen["air_mass"].fillna(0).mean()))
    assert np.isclose(float(gen["air_mass"].fillna(0).mean()), 3.9380521870930165)
    print(float(gen["direct_normal_irradiance"].fillna(0).mean()))
    assert np.isclose(
        float(gen["direct_normal_irradiance"].fillna(0).mean()), 20.907640632171837
    )
    print(float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()))
    assert np.isclose(
        float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()), 19.589003059825554
    )
    print(float(gen["angle_of_incidence"].fillna(0).mean()))
    assert np.isclose(
        float(gen["angle_of_incidence"].fillna(0).mean()), 18.915157478877003
    )
    assert np.isclose(float(gen["poa_global"].fillna(0).mean()), 38.581645380369565)
    assert np.isclose(float(gen["poa_direct"].fillna(0).mean()), 15.491403834596987)
    assert np.isclose(float(gen["poa_diffuse"].fillna(0).mean()), 23.090241545772578)
    assert np.isclose(float(gen["poa_sky_diffuse"].fillna(0).mean()), 22.5137080215302)
    assert np.isclose(
        float(gen["poa_ground_diffuse"].fillna(0).mean()), 0.5765335242423779
    )
    assert np.isclose(
        float(gen["cell_temperature"].fillna(0).mean()), 1.757604391183729
    )
    assert np.isclose(
        float(gen["module_dc_power_at_mpp"].fillna(0).mean()), 12.625206716607272
    )
    assert np.isclose(
        float(gen["module_dc_voltage_at_mpp"].fillna(0).mean()), 14.060960010572025
    )
    assert np.isclose(
        float(gen["capacity_factor"].fillna(0).mean()), 0.04201539723986579
    )
    assert np.isclose(
        float(gen["total_system_generation"].fillna(0).mean()), 84.0307944797316
    )


def test_openfield_pv_sarah_unvalidated(pt_pv_placements):
    gen = openfield_pv_sarah_unvalidated(
        placements=pt_pv_placements,
        sarah_path=TEST_DATA["sarah-like"],
        era5_path=TEST_DATA["era5-like"],
        module="WINAICO WSx-240P6",
        elev=300,
        tracking="fixed",
        inverter=None,
        inverter_kwargs={},
        tracking_args={},
        output_netcdf_path=None,
        output_variables=None,
    )

    assert gen["location"].shape == (560,)
    assert gen["capacity"].shape == (560,)
    assert gen["lon"].shape == (560,)
    assert gen["lat"].shape == (560,)
    assert gen["tilt"].shape == (560,)
    assert gen["azimuth"].shape == (560,)
    assert gen["elev"].shape == (560,)
    assert gen["time"].shape == (48,)
    assert gen["direct_normal_irradiance"].shape == (48, 560)
    assert gen["global_horizontal_irradiance"].shape == (48, 560)
    assert gen["surface_wind_speed"].shape == (48, 560)
    assert gen["surface_pressure"].shape == (48, 560)
    assert gen["surface_air_temperature"].shape == (48, 560)
    assert gen["surface_dew_temperature"].shape == (48, 560)
    assert gen["solar_azimuth"].shape == (48, 560)
    assert gen["apparent_solar_zenith"].shape == (48, 560)
    assert gen["extra_terrestrial_irradiance"].shape == (48, 560)
    assert gen["air_mass"].shape == (48, 560)
    assert gen["diffuse_horizontal_irradiance"].shape == (48, 560)
    assert gen["angle_of_incidence"].shape == (48, 560)
    assert gen["poa_global"].shape == (48, 560)
    assert gen["poa_direct"].shape == (48, 560)
    assert gen["poa_diffuse"].shape == (48, 560)
    assert gen["poa_sky_diffuse"].shape == (48, 560)
    assert gen["poa_ground_diffuse"].shape == (48, 560)
    assert gen["cell_temperature"].shape == (48, 560)
    assert gen["module_dc_power_at_mpp"].shape == (48, 560)
    assert gen["module_dc_voltage_at_mpp"].shape == (48, 560)
    assert gen["capacity_factor"].shape == (48, 560)
    assert gen["total_system_generation"].shape == (48, 560)

    # assert np.isclose( float(gen['location'].fillna(0).mean() ), 279.5)
    assert np.isclose(float(gen["location"].fillna(0).mean()), 279.5)
    assert np.isclose(float(gen["capacity"].fillna(0).mean()), 2000.0)
    assert np.isclose(float(gen["lon"].fillna(0).mean()), 6.16945196229404)
    assert np.isclose(float(gen["lat"].fillna(0).mean()), 50.80320853112445)
    assert np.isclose(float(gen["tilt"].fillna(0).mean()), 39.19976325987092)
    assert np.isclose(float(gen["azimuth"].fillna(0).mean()), 180.0)
    assert np.isclose(float(gen["elev"].fillna(0).mean()), 300.0)
    assert np.isclose(
        float(gen["direct_normal_irradiance"].fillna(0).mean()), 155.98687394203432
    )
    assert np.isclose(
        float(gen["global_horizontal_irradiance"].fillna(0).mean()), 50.013295986799676
    )
    assert np.isclose(
        float(gen["surface_wind_speed"].fillna(0).mean()), 1.7504422159388175
    )
    assert np.isclose(
        float(gen["surface_pressure"].fillna(0).mean()), 37772.375143895624
    )
    assert np.isclose(
        float(gen["surface_air_temperature"].fillna(0).mean()), 0.9744363284867832
    )
    assert np.isclose(
        float(gen["surface_dew_temperature"].fillna(0).mean()), -0.3616572402515169
    )
    assert np.isclose(float(gen["solar_azimuth"].fillna(0).mean()), 68.01052582121625)
    assert np.isclose(
        float(gen["apparent_solar_zenith"].fillna(0).mean()), 30.37642178304629
    )
    assert np.isclose(
        float(gen["extra_terrestrial_irradiance"].fillna(0).mean()), 531.7569374999998
    )
    assert np.isclose(float(gen["air_mass"].fillna(0).mean()), 3.7818783820335424)
    assert np.isclose(
        float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()), 16.019586491017332
    )
    assert np.isclose(
        float(gen["angle_of_incidence"].fillna(0).mean()), 18.703054670614844
    )
    assert np.isclose(float(gen["poa_global"].fillna(0).mean()), 140.8677773815084)
    assert np.isclose(float(gen["poa_direct"].fillna(0).mean()), 112.10492816869589)
    assert np.isclose(float(gen["poa_diffuse"].fillna(0).mean()), 28.762849212812483)
    assert np.isclose(
        float(gen["poa_sky_diffuse"].fillna(0).mean()), 27.582410926737612
    )
    assert np.isclose(
        float(gen["poa_ground_diffuse"].fillna(0).mean()), 1.180438286074874
    )
    assert np.isclose(
        float(gen["cell_temperature"].fillna(0).mean()), 4.738007290882006
    )
    assert np.isclose(
        float(gen["module_dc_power_at_mpp"].fillna(0).mean()), 45.17960123200836
    )
    assert np.isclose(
        float(gen["module_dc_voltage_at_mpp"].fillna(0).mean()), 11.940731512724874
    )
    assert np.isclose(
        float(gen["capacity_factor"].fillna(0).mean()), 0.15035309405307454
    )
    assert np.isclose(
        float(gen["total_system_generation"].fillna(0).mean()), 300.70618810614906
    )
