import geokit as gk
import numpy as np
import pandas as pd
import pytest

from reskit import data
from reskit.solar.workflows.workflows import (
    openfield_pv_era5,
    openfield_pv_iconlam,
    openfield_pv_merra_ryberg2019,
    openfield_pv_sarah_unvalidated,
)

FIXTURES = data.paths("test_suite")


@pytest.fixture
def pt_pv_placements() -> pd.DataFrame:
    df = gk.vector.extractFeatures(FIXTURES["turbine_placements_shp"])
    df["capacity"] = 2000
    return df


@pytest.fixture
def pt_pv_placements_Zimbabwe() -> pd.DataFrame:
    # Keep numerical regression inputs independent of administrative boundary updates.
    # The 2025 tables: their provenance is in the bundled description of
    # reskit-test-data/placements.
    df = pd.read_csv(FIXTURES["module_placements_bulawayo"])

    return df


def test_openfield_pv_iconlam(pt_pv_placements_Zimbabwe):
    gen = openfield_pv_iconlam(
        placements=pt_pv_placements_Zimbabwe,
        icon_lam_path=FIXTURES["icon_lam"],
        module="WINAICO WSx-240P6",
        elev=300,
        tracking="fixed",
        inverter=None,
        inverter_kwargs={},
        tracking_args={},
        output_netcdf_path=None,
        output_variables=None,
        tech_year=2050,
    )

    assert gen["location"].shape == (483,)
    assert gen["capacity"].shape == (483,)
    assert gen["lon"].shape == (483,)
    assert gen["lat"].shape == (483,)
    assert gen["tilt"].shape == (483,)
    assert gen["azimuth"].shape == (483,)
    assert gen["elev"].shape == (483,)
    assert gen["time"].shape == (144,)
    assert gen["global_horizontal_irradiance"].shape == (144, 483)
    assert gen["direct_horizontal_irradiance"].shape == (144, 483)
    assert gen["surface_wind_speed"].shape == (144, 483)
    assert gen["surface_pressure"].shape == (144, 483)
    assert gen["surface_air_temperature"].shape == (144, 483)
    assert gen["surface_dew_temperature"].shape == (144, 483)
    assert gen["solar_azimuth"].shape == (144, 483)
    assert gen["apparent_solar_zenith"].shape == (144, 483)
    assert gen["direct_normal_irradiance"].shape == (144, 483)
    assert gen["extra_terrestrial_irradiance"].shape == (144, 483)
    assert gen["air_mass"].shape == (144, 483)
    assert gen["diffuse_horizontal_irradiance"].shape == (144, 483)
    assert gen["angle_of_incidence"].shape == (144, 483)
    assert gen["poa_global"].shape == (144, 483)
    assert gen["poa_direct"].shape == (144, 483)
    assert gen["poa_diffuse"].shape == (144, 483)
    assert gen["poa_sky_diffuse"].shape == (144, 483)
    assert gen["poa_ground_diffuse"].shape == (144, 483)
    assert gen["cell_temperature"].shape == (144, 483)
    assert gen["module_dc_power_at_mpp"].shape == (144, 483)
    assert gen["module_dc_voltage_at_mpp"].shape == (144, 483)
    assert gen["capacity_factor"].shape == (144, 483)
    assert gen["total_system_generation"].shape == (144, 483)

    assert np.isclose(float(gen["location"].fillna(0).mean()), 241.0)
    assert np.isclose(float(gen["capacity"].fillna(0).mean()), 14533.953933747413)
    assert np.isclose(float(gen["lon"].fillna(0).mean()), 28.50841382368852)
    assert np.isclose(float(gen["lat"].fillna(0).mean()), -20.117193185204783)
    assert np.isclose(float(gen["tilt"].fillna(0).mean()), 20.518451664897075)
    assert np.isclose(float(gen["azimuth"].fillna(0).mean()), 0.0)
    assert np.isclose(float(gen["elev"].fillna(0).mean()), 300.0)
    assert np.isclose(float(gen["global_horizontal_irradiance"].fillna(0).mean()), 345.6062589173662)
    assert np.isclose(float(gen["direct_horizontal_irradiance"].fillna(0).mean()), 279.838384668487)
    assert np.isclose(float(gen["surface_wind_speed"].fillna(0).mean()), 1.564691383184658)
    assert np.isclose(float(gen["surface_pressure"].fillna(0).mean()), 51468.50971934668)
    assert np.isclose(float(gen["surface_air_temperature"].fillna(0).mean()), 15.760231596369948)
    assert np.isclose(float(gen["surface_dew_temperature"].fillna(0).mean()), 8.838034817836245)
    assert np.isclose(float(gen["solar_azimuth"].fillna(0).mean()), 91.82659189011802)
    assert np.isclose(float(gen["apparent_solar_zenith"].fillna(0).mean()), 28.17693142540546)
    assert np.isclose(float(gen["direct_normal_irradiance"].fillna(0).mean()), 357.85820433247335)
    assert np.isclose(float(gen["extra_terrestrial_irradiance"].fillna(0).mean()), 823.8760486889863)
    assert np.isclose(float(gen["air_mass"].fillna(0).mean()), 2.531118905657518)
    assert np.isclose(float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()), 66.2378169320506)
    assert np.isclose(float(gen["angle_of_incidence"].fillna(0).mean()), 28.082660955171594)
    assert np.isclose(float(gen["poa_global"].fillna(0).mean()), 343.23936122056716)
    assert np.isclose(float(gen["poa_direct"].fillna(0).mean()), 276.55530243973722)
    assert np.isclose(float(gen["poa_diffuse"].fillna(0).mean()), 66.68405878082996)
    assert np.isclose(float(gen["poa_sky_diffuse"].fillna(0).mean()), 64.8764543373828)
    assert np.isclose(float(gen["poa_ground_diffuse"].fillna(0).mean()), 1.8076044434471568)
    assert np.isclose(float(gen["cell_temperature"].fillna(0).mean()), 25.86583280350363)
    assert np.isclose(float(gen["module_dc_power_at_mpp"].fillna(0).mean()), 94.00160119470556)
    assert np.isclose(float(gen["module_dc_voltage_at_mpp"].fillna(0).mean()), 18.773821261251605)
    assert np.isclose(float(gen["capacity_factor"].fillna(0).mean()), 0.34919394100831996)
    assert np.isclose(float(gen["total_system_generation"].fillna(0).mean()), 5082.087226375295)


def test_openfield_pv_era5(pt_pv_placements):
    gen = openfield_pv_era5(
        placements=pt_pv_placements,
        era5_path=FIXTURES["era5"],
        global_solar_atlas_ghi_path=FIXTURES["gsa_ghi"],
        global_solar_atlas_dni_path=FIXTURES["gsa_dni"],
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
    assert np.isclose(float(gen["global_horizontal_irradiance"].fillna(0).mean()), 32.90016155215698)
    assert np.isclose(float(gen["direct_horizontal_irradiance"].fillna(0).mean()), 15.501608137870793)
    assert np.isclose(float(gen["surface_wind_speed"].fillna(0).mean()), 1.6521243123091525)
    assert np.isclose(float(gen["surface_pressure"].fillna(0).mean()), 38644.083559948376)
    assert np.isclose(float(gen["surface_air_temperature"].fillna(0).mean()), 1.0433187770747245)
    assert np.isclose(float(gen["surface_dew_temperature"].fillna(0).mean()), 0.014244860844314216)
    assert np.isclose(float(gen["solar_azimuth"].fillna(0).mean()), 68.6008947378997)
    assert np.isclose(float(gen["apparent_solar_zenith"].fillna(0).mean()), 31.14453235068293)
    assert np.isclose(float(gen["direct_normal_irradiance"].fillna(0).mean()), 51.13683382510634)
    assert np.isclose(float(gen["extra_terrestrial_irradiance"].fillna(0).mean()), 546.9559617849145)
    assert np.isclose(float(gen["air_mass"].fillna(0).mean()), 3.910682025984337)
    assert np.isclose(float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()), 20.957345453247044)
    assert np.isclose(float(gen["angle_of_incidence"].fillna(0).mean()), 19.14992962052318)
    assert np.isclose(float(gen["poa_global"].fillna(0).mean()), 67.12953428476871)
    assert np.isclose(float(gen["poa_direct"].fillna(0).mean()), 37.864288786242795)
    assert np.isclose(float(gen["poa_diffuse"].fillna(0).mean()), 29.265245498525918)
    assert np.isclose(float(gen["poa_sky_diffuse"].fillna(0).mean()), 28.48874351889887)
    assert np.isclose(float(gen["poa_ground_diffuse"].fillna(0).mean()), 0.7765019796270453)
    assert np.isclose(float(gen["cell_temperature"].fillna(0).mean()), 2.9014410638713493)
    assert np.isclose(float(gen["module_dc_power_at_mpp"].fillna(0).mean()), 21.8408291970829)
    assert np.isclose(float(gen["module_dc_voltage_at_mpp"].fillna(0).mean()), 14.348350493615033)
    assert np.isclose(float(gen["capacity_factor"].fillna(0).mean()), 0.08040672667733686)
    assert np.isclose(float(gen["total_system_generation"].fillna(0).mean()), 160.81345335467375)


def test_openfield_pv_merra_ryberg2019(pt_pv_placements):
    gen = openfield_pv_merra_ryberg2019(
        placements=pt_pv_placements,
        merra_path=FIXTURES["merra"],
        global_solar_atlas_ghi_path=FIXTURES["gsa_ghi"],
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
    assert np.isclose(float(gen["surface_wind_speed"].fillna(0).mean()), 1.5502203948117972)
    print(float(gen["surface_pressure"].fillna(0).mean()))
    assert np.isclose(float(gen["surface_pressure"].fillna(0).mean()), 38110.883667100796)
    print(float(gen["surface_air_temperature"].fillna(0).mean()))
    assert np.isclose(float(gen["surface_air_temperature"].fillna(0).mean()), 0.6923904404714382)
    print(float(gen["surface_dew_temperature"].fillna(0).mean()))
    assert np.isclose(float(gen["surface_dew_temperature"].fillna(0).mean()), 0.2735079282721086)
    print(float(gen["global_horizontal_irradiance"].fillna(0).mean()))
    assert np.isclose(float(gen["global_horizontal_irradiance"].fillna(0).mean()), 24.425654064650278)
    print(float(gen["solar_azimuth"].fillna(0).mean()))
    assert np.isclose(float(gen["solar_azimuth"].fillna(0).mean()), 67.69226199649943)
    print(float(gen["apparent_solar_zenith"].fillna(0).mean()))
    assert np.isclose(float(gen["apparent_solar_zenith"].fillna(0).mean()), 30.756281891950543)
    print(float(gen["extra_terrestrial_irradiance"].fillna(0).mean()))
    assert np.isclose(float(gen["extra_terrestrial_irradiance"].fillna(0).mean()), 539.2545578051567)
    print(float(gen["air_mass"].fillna(0).mean()))
    assert np.isclose(float(gen["air_mass"].fillna(0).mean()), 3.9389963168140474)
    print(float(gen["direct_normal_irradiance"].fillna(0).mean()))
    assert np.isclose(float(gen["direct_normal_irradiance"].fillna(0).mean()), 20.940813999631725)
    print(float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()))
    assert np.isclose(float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()), 19.584935193047343)
    print(float(gen["angle_of_incidence"].fillna(0).mean()))
    assert np.isclose(float(gen["angle_of_incidence"].fillna(0).mean()), 18.916229003656685)
    assert np.isclose(float(gen["poa_global"].fillna(0).mean()), 38.605102528112646)
    assert np.isclose(float(gen["poa_direct"].fillna(0).mean()), 15.51131994758096)
    assert np.isclose(float(gen["poa_diffuse"].fillna(0).mean()), 23.093782580531688)
    assert np.isclose(float(gen["poa_sky_diffuse"].fillna(0).mean()), 22.5172490562893)
    assert np.isclose(float(gen["poa_ground_diffuse"].fillna(0).mean()), 0.5765335242423779)
    assert np.isclose(float(gen["cell_temperature"].fillna(0).mean()), 1.758253693370065)
    assert np.isclose(float(gen["module_dc_power_at_mpp"].fillna(0).mean()), 12.632992225468614)
    assert np.isclose(float(gen["module_dc_voltage_at_mpp"].fillna(0).mean()), 14.059241131371758)
    assert np.isclose(float(gen["capacity_factor"].fillna(0).mean()), 0.04204130661742026)
    assert np.isclose(float(gen["total_system_generation"].fillna(0).mean()), 84.08261323484052)


def test_openfield_pv_sarah_unvalidated(pt_pv_placements):
    gen = openfield_pv_sarah_unvalidated(
        placements=pt_pv_placements,
        sarah_path=FIXTURES["sarah"],
        era5_path=FIXTURES["era5"],
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
    assert np.isclose(float(gen["direct_normal_irradiance"].fillna(0).mean()), 155.98687394203432)
    assert np.isclose(float(gen["global_horizontal_irradiance"].fillna(0).mean()), 50.013295986799676)
    assert np.isclose(float(gen["surface_wind_speed"].fillna(0).mean()), 1.7504422159388175)
    assert np.isclose(float(gen["surface_pressure"].fillna(0).mean()), 37772.375143895624)
    assert np.isclose(float(gen["surface_air_temperature"].fillna(0).mean()), 0.9744363284867832)
    assert np.isclose(float(gen["surface_dew_temperature"].fillna(0).mean()), -0.3616572402515169)
    assert np.isclose(float(gen["solar_azimuth"].fillna(0).mean()), 68.01052582121625)
    assert np.isclose(float(gen["apparent_solar_zenith"].fillna(0).mean()), 30.377618465445458)
    assert np.isclose(float(gen["extra_terrestrial_irradiance"].fillna(0).mean()), 531.7569374999998)
    assert np.isclose(float(gen["air_mass"].fillna(0).mean()), 3.7828382694554725)
    assert np.isclose(float(gen["diffuse_horizontal_irradiance"].fillna(0).mean()), 16.029016338165572)
    assert np.isclose(float(gen["angle_of_incidence"].fillna(0).mean()), 18.704099313472938)
    assert np.isclose(float(gen["poa_global"].fillna(0).mean()), 140.87033904207777)
    assert np.isclose(float(gen["poa_direct"].fillna(0).mean()), 112.09877622231873)
    assert np.isclose(float(gen["poa_diffuse"].fillna(0).mean()), 28.771562819759)
    assert np.isclose(float(gen["poa_sky_diffuse"].fillna(0).mean()), 27.591124533684138)
    assert np.isclose(float(gen["poa_ground_diffuse"].fillna(0).mean()), 1.180438286074874)
    assert np.isclose(float(gen["cell_temperature"].fillna(0).mean()), 4.738078495069038)
    assert np.isclose(float(gen["module_dc_power_at_mpp"].fillna(0).mean()), 45.18028935624859)
    assert np.isclose(float(gen["module_dc_voltage_at_mpp"].fillna(0).mean()), 11.939309430641291)
    assert np.isclose(float(gen["capacity_factor"].fillna(0).mean()), 0.150355384060197)
    assert np.isclose(float(gen["total_system_generation"].fillna(0).mean()), 300.71076812039394)
