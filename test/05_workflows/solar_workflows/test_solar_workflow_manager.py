import pandas as pd
import numpy as np
from reskit.solar import SolarWorkflowManager
import reskit as rk
import geokit as gk
import pytest


def print_testresults(variable):
    print("mean: ", variable[0:140, :].mean())
    print("std: ", variable[0:140, :].std())
    print("min: ", variable[0:140, :].min())
    print("max: ", variable[0:140, :].max())


def test_SolarWorkflowManager___init__() -> SolarWorkflowManager:
    # (self, placements):
    placements = pd.DataFrame()
    placements["lon"] = [
        6.083,
        6.183,
        6.083,
        6.183,
        6.083,
    ]
    placements["lat"] = [
        50.475,
        50.575,
        50.675,
        50.775,
        50.875,
    ]
    placements["capacity"] = [
        2000,
        2500,
        3000,
        3500,
        4000,
    ]
    placements["tilt"] = [
        20,
        25,
        30,
        35,
        40,
    ]
    placements["azimuth"] = [180, 180, 180, 180, 180]

    man = SolarWorkflowManager(placements)

    assert np.isclose(man.ext.xMin, 6.083000)
    assert np.isclose(man.ext.xMax, 6.183000)
    assert np.isclose(man.ext.yMin, 50.475000)
    assert np.isclose(man.ext.yMax, 50.875000)

    assert (man.placements["lon"] == placements["lon"]).all()
    assert (man.placements["lat"] == placements["lat"]).all()
    assert (man.placements["capacity"] == placements["capacity"]).all()
    assert (man.placements["tilt"] == placements["tilt"]).all()
    assert (man.placements["azimuth"] == placements["azimuth"]).all()

    return man


@pytest.fixture
def pt_SolarWorkflowManager_initialized() -> SolarWorkflowManager:
    return test_SolarWorkflowManager___init__()


def test_SolarWorkflowManager_estimate_tilt_from_latitude(
    pt_SolarWorkflowManager_initialized,
):
    # (self, convention):
    man = pt_SolarWorkflowManager_initialized

    man.estimate_tilt_from_latitude("Ryberg2020")

    assert np.isclose(
        man.placements["tilt"],
        [39.0679049, 39.1082060, 39.1484058, 39.1885045, 39.2285025],
    ).all()

    man.estimate_tilt_from_latitude("(latitude-5)**2")

    assert np.isclose(
        man.placements["tilt"],
        [2067.975625, 2077.080625, 2086.205625, 2095.350625, 2104.515625],
    ).all()


def test_SolarWorkflowManager_estimate_azimuth_from_latitude(
    pt_SolarWorkflowManager_initialized,
):
    man = pt_SolarWorkflowManager_initialized

    man.estimate_azimuth_from_latitude()

    assert np.isclose(man.placements["azimuth"], [180, 180, 180, 180, 180]).all()

    man.placements["lat"] *= -1
    man.locs = gk.LocationSet(man.placements[["lon", "lat"]].values)
    man.estimate_azimuth_from_latitude()

    assert np.isclose(man.placements["azimuth"], [0, 0, 0, 0, 0]).all()


def test_SolarWorkflowManager_apply_elevation(pt_SolarWorkflowManager_initialized):
    man = pt_SolarWorkflowManager_initialized

    # first test None case without elev attribute in placements
    man.apply_elevation(elev=None, fallback_elev=-1000)
    # must yield fallback value for all locations
    assert np.isclose(
        man.placements["elev"], [-1000, -1000, -1000, -1000, -1000]
    ).all()

    # now test using the elevation from the placements dataframe
    base_elev = [90, 80, 70, 60, 50]
    man.placements["elev"] = base_elev
    man.apply_elevation(elev=None, fallback_elev=-1000)
    # the elev data must not have been altered when None and 'elev' in attribute
    assert np.isclose(man.placements["elev"], base_elev).all()

    # then test scalar value
    man.apply_elevation(elev=120, fallback_elev=-1000)
    # must yield this value for all locs
    assert np.isclose(man.placements["elev"], [120, 120, 120, 120, 120]).all()

    # next test iterable as new elev
    new_elev = [100, 120, 140, 160, 2000]
    man.apply_elevation(elev=new_elev, fallback_elev=-1000)
    # must yield the same iterable
    assert np.isclose(man.placements["elev"], new_elev).all()

    # last test raster elevation
    man.apply_elevation(
        elev=rk.TEST_DATA["clc-aachen_clipped.tif"], fallback_elev=-1000
    )  # not an elevation file, but still a raster
    # must yield raster values, with fallback value for those placements outside the actual file coverage
    assert np.isclose(
        man.placements["elev"],
        # TODO these values were from rk.TEST_DATA['gwa50-like.tif'], adapt to CLC values and fallbacks depending on which are outside
        [2, 36, 18, 18, 21],
    ).all()

    return man


@pytest.fixture
def pt_SolarWorkflowManager_loaded(
    pt_SolarWorkflowManager_initialized: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_initialized
    man.apply_elevation([100, 120, 140, 160, 2000])

    man.read(
        variables=[
            "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
        ],
        source_type="ERA5",
        source=rk.TEST_DATA["era5-like"],
        set_time_index=True,
        verbose=False,
    )

    return man


def test_SolarWorkflowManager_determine_solar_position(
    pt_SolarWorkflowManager_loaded: SolarWorkflowManager,
) -> SolarWorkflowManager:
    # (self, lon_rounding=1, lat_rounding=1, elev_rounding=-2):
    man = pt_SolarWorkflowManager_loaded

    man.determine_solar_position(
        lon_rounding=1,
        lat_rounding=1,
        elev_rounding=-2,
    )

    assert man.sim_data["solar_azimuth"].shape == (140, 5)

    assert np.isclose(man.sim_data["solar_azimuth"].mean(), 181.75084452775852)
    assert np.isclose(man.sim_data["solar_azimuth"].std(), 90.18959294069582)
    assert np.isclose(man.sim_data["solar_azimuth"].min(), 23.100292572931437)
    assert np.isclose(man.sim_data["solar_azimuth"].max(), 355.8650905234781)

    assert np.isclose(man.sim_data["apparent_solar_zenith"].mean(), 108.93266465908583)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].std(), 26.914599770957278)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].min(), 72.98977919840057)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].max(), 152.49005970814673)

    return man


@pytest.fixture
def pt_SolarWorkflowManager_solpos(
    pt_SolarWorkflowManager_loaded: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded

    man.determine_solar_position(
        lon_rounding=1,
        lat_rounding=1,
        elev_rounding=-2,
    )

    return man


def test_SolarWorkflowManager_filter_positive_solar_elevation(
    pt_SolarWorkflowManager_solpos: SolarWorkflowManager,
) -> SolarWorkflowManager:
    # (self):
    man = pt_SolarWorkflowManager_solpos

    man.filter_positive_solar_elevation()

    print_testresults(man.sim_data["solar_azimuth"])
    print_testresults(man.sim_data["apparent_solar_zenith"])

    assert man.sim_data["solar_azimuth"].shape == (54, 5)
    assert np.isclose(man.sim_data["solar_azimuth"].mean(), 177.8281611330465)
    assert np.isclose(man.sim_data["solar_azimuth"].std(), 34.898695009531934)
    assert np.isclose(man.sim_data["solar_azimuth"].min(), 124.71102348726265)
    assert np.isclose(man.sim_data["solar_azimuth"].max(), 231.1922838037285)

    assert np.isclose(man.sim_data["apparent_solar_zenith"].mean(), 80.66303691482851)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].std(), 6.193123877473528)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].min(), 72.98977919840057)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].max(), 91.89378124249767)

    return man


def test_SolarWorkflowManager_determine_extra_terrestrial_irradiance(
    pt_SolarWorkflowManager_solpos: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_solpos
    man.determine_extra_terrestrial_irradiance()

    print_testresults(man.sim_data["extra_terrestrial_irradiance"])

    assert man.sim_data["extra_terrestrial_irradiance"].shape == (140, 5)
    assert np.isclose(
        man.sim_data["extra_terrestrial_irradiance"].mean(), 1413.9980694079702
    )
    assert np.isclose(
        man.sim_data["extra_terrestrial_irradiance"].std(), 0.019625866056578487
    )
    assert np.isclose(
        man.sim_data["extra_terrestrial_irradiance"].min(), 1413.940576307916
    )
    assert np.isclose(
        man.sim_data["extra_terrestrial_irradiance"].max(), 1414.0192010311885
    )

    return man


def test_SolarWorkflowManager_determine_air_mass(
    pt_SolarWorkflowManager_solpos: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_solpos
    man.determine_air_mass(model="kastenyoung1989")

    print_testresults(man.sim_data["air_mass"])

    assert man.sim_data["air_mass"].shape == (140, 5)
    assert np.isclose(man.sim_data["air_mass"].mean(), 21.689624010649034)
    assert np.isclose(man.sim_data["air_mass"].std(), 10.849130623014739)
    assert np.isclose(man.sim_data["air_mass"].min(), 3.383950740640421)
    assert np.isclose(man.sim_data["air_mass"].max(), 29.0)

    return man


@pytest.fixture
def pt_SolarWorkflowManager_loaded2(
    pt_SolarWorkflowManager_solpos: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_solpos
    man.filter_positive_solar_elevation()
    man.determine_extra_terrestrial_irradiance()
    man.determine_air_mass(model="kastenyoung1989")

    return man


def test_SolarWorkflowManager_apply_DIRINT_model(
    pt_SolarWorkflowManager_loaded2: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded2
    man.apply_DIRINT_model(use_pressure=True, use_dew_temperature=True)

    print_testresults(man.sim_data["direct_normal_irradiance"])

    assert man.sim_data["direct_normal_irradiance"].shape == (54, 5)
    assert np.isclose(
        man.sim_data["direct_normal_irradiance"].mean(), 167.86780412863015
    )
    assert np.isclose(
        man.sim_data["direct_normal_irradiance"].std(), 202.51729861336193
    )
    assert np.isclose(man.sim_data["direct_normal_irradiance"].min(), 0.0)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].max(), 720.1159360124137)


@pytest.fixture
def pt_SolarWorkflowManager_dni(
    pt_SolarWorkflowManager_loaded2: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded2
    man.apply_DIRINT_model(use_pressure=True, use_dew_temperature=True)

    return man


def test_SolarWorkflowManager_diffuse_horizontal_irradiance_from_trigonometry(
    pt_SolarWorkflowManager_dni: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_dni
    man.diffuse_horizontal_irradiance_from_trigonometry()

    print_testresults(man.sim_data["diffuse_horizontal_irradiance"])

    assert man.sim_data["diffuse_horizontal_irradiance"].shape == (54, 5)
    assert np.isclose(
        man.sim_data["diffuse_horizontal_irradiance"].mean(), 48.582931923941324
    )
    assert np.isclose(
        man.sim_data["diffuse_horizontal_irradiance"].std(), 34.69121106889705
    )
    assert np.isclose(
        man.sim_data["diffuse_horizontal_irradiance"].min(), 0.15659047212134164
    )
    assert np.isclose(
        man.sim_data["diffuse_horizontal_irradiance"].max(), 124.98184251575456
    )


def test_SolarWorkflowManager_direct_normal_irradiance_from_trigonometry(
    pt_SolarWorkflowManager_loaded2: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded2

    man.direct_normal_irradiance_from_trigonometry()

    print_testresults(man.sim_data["direct_normal_irradiance"])

    assert man.sim_data["direct_normal_irradiance"].shape == (54, 5)
    assert np.isclose(
        man.sim_data["direct_normal_irradiance"].mean(), 158.21469197801994
    )
    assert np.isclose(man.sim_data["direct_normal_irradiance"].std(), 179.6328322092467)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].min(), 0.0)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].max(), 616.5611489924958)


@pytest.fixture
def pt_SolarWorkflowManager_all_irrad(
    pt_SolarWorkflowManager_loaded2: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded2
    # man = pt_SolarWorkflowManager_dni
    man.direct_normal_irradiance_from_trigonometry()
    man.diffuse_horizontal_irradiance_from_trigonometry()

    return man


def test_SolarWorkflowManager_permit_single_axis_tracking(
    pt_SolarWorkflowManager_all_irrad: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_all_irrad
    man.permit_single_axis_tracking(
        max_angle=90,
        backtrack=True,
        gcr=0.2857142857142857,
    )

    print_testresults(man.sim_data["system_tilt"])
    print_testresults(man.sim_data["system_azimuth"])

    assert man.sim_data["system_tilt"].shape == (54, 5)
    assert np.isclose(man.sim_data["system_tilt"].mean(), 46.36795184688052)
    assert np.isclose(man.sim_data["system_tilt"].std(), 14.570819765672116)
    assert np.isclose(man.sim_data["system_tilt"].min(), 20.0)
    assert np.isclose(man.sim_data["system_tilt"].max(), 74.30021518311098)

    assert np.isclose(man.sim_data["system_azimuth"].mean(), 185.84603169500053)
    assert np.isclose(man.sim_data["system_azimuth"].std(), 52.78835501687092)
    assert np.isclose(man.sim_data["system_azimuth"].min(), 99.71477147193693)
    assert np.isclose(man.sim_data["system_azimuth"].max(), 264.12802748241154)


def test_SolarWorkflowManager_determine_angle_of_incidence(
    pt_SolarWorkflowManager_all_irrad: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_all_irrad
    man.determine_angle_of_incidence()

    print_testresults(man.sim_data["angle_of_incidence"])

    assert man.sim_data["angle_of_incidence"].shape == (54, 5)
    assert np.isclose(man.sim_data["angle_of_incidence"].mean(), 56.59448446923573)
    assert np.isclose(man.sim_data["angle_of_incidence"].std(), 12.022866054675205)
    assert np.isclose(man.sim_data["angle_of_incidence"].min(), 33.45888119832759)
    assert np.isclose(man.sim_data["angle_of_incidence"].max(), 80.25014934148591)


@pytest.fixture
def pt_SolarWorkflowManager_aoi(
    pt_SolarWorkflowManager_all_irrad: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_all_irrad
    man.determine_angle_of_incidence()

    return man


def test_SolarWorkflowManager_estimate_plane_of_array_irradiances(
    pt_SolarWorkflowManager_aoi: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_aoi
    man.estimate_plane_of_array_irradiances(
        transposition_model="perez",
    )

    print_testresults(man.sim_data["poa_global"])
    print(man.sim_data["poa_direct"].mean())
    print(man.sim_data["poa_diffuse"].mean())
    print(man.sim_data["poa_sky_diffuse"].mean())
    print(man.sim_data["poa_ground_diffuse"].mean())

    assert man.sim_data["poa_global"].shape == (54, 5)

    assert np.isclose(man.sim_data["poa_global"].mean(), 174.11992196172187)
    assert np.isclose(man.sim_data["poa_global"].std(), 173.4474037663958)
    assert np.isclose(man.sim_data["poa_global"].min(), 0.13328509297399485)
    assert np.isclose(man.sim_data["poa_global"].max(), 621.2447325355588)

    assert np.isclose(man.sim_data["poa_direct"].mean(), 102.74712287621118)
    assert np.isclose(man.sim_data["poa_diffuse"].mean(), 71.37279908551066)
    assert np.isclose(man.sim_data["poa_sky_diffuse"].mean(), 69.85080250223847)
    assert np.isclose(man.sim_data["poa_ground_diffuse"].mean(), 1.52199658327221)


@pytest.fixture
def pt_SolarWorkflowManager_poa(
    pt_SolarWorkflowManager_aoi: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_aoi
    man.estimate_plane_of_array_irradiances(transposition_model="perez", albedo=0.25)

    return man


def test_SolarWorkflowManager_cell_temperature_from_sapm(
    pt_SolarWorkflowManager_poa: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa

    man.cell_temperature_from_sapm(mounting="glass_open_rack")

    print_testresults(man.sim_data["cell_temperature"])

    assert man.sim_data["cell_temperature"].shape == (54, 5)
    assert np.isclose(man.sim_data["cell_temperature"].mean(), 6.700896196088481)
    assert np.isclose(man.sim_data["cell_temperature"].std(), 5.647128705200129)
    assert np.isclose(man.sim_data["cell_temperature"].min(), -3.2822952246943804)
    assert np.isclose(man.sim_data["cell_temperature"].max(), 21.181626183824648)

    # roof top PV should run hotter than open-field
    man.cell_temperature_from_sapm(mounting="glass_close_roof")

    print_testresults(man.sim_data["cell_temperature"])

    assert man.sim_data["cell_temperature"].shape == (54, 5)
    assert np.isclose(man.sim_data["cell_temperature"].mean(), 9.406303607095017)
    assert np.isclose(man.sim_data["cell_temperature"].std(), 8.25669361128076)
    assert np.isclose(man.sim_data["cell_temperature"].min(), -3.2472615808752097)
    assert np.isclose(man.sim_data["cell_temperature"].max(), 31.095795267573923)


def test_SolarWorkflowManager_apply_angle_of_incidence_losses_to_poa(
    pt_SolarWorkflowManager_poa: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa
    man.apply_angle_of_incidence_losses_to_poa()

    print_testresults(man.sim_data["poa_global"])
    assert man.sim_data["poa_global"].shape == (54, 5)
    assert np.isclose(man.sim_data["poa_global"].mean(), 168.7581881488339)
    assert np.isclose(man.sim_data["poa_global"].std(), 169.56605012489317)
    assert np.isclose(man.sim_data["poa_global"].min(), 0.12759789566504143)
    assert np.isclose(man.sim_data["poa_global"].max(), 613.4267695866687)

    print(man.sim_data["poa_direct"].mean())
    print(man.sim_data["poa_diffuse"].mean())
    print(man.sim_data["poa_sky_diffuse"].mean())
    print(man.sim_data["poa_ground_diffuse"].mean())
    assert np.isclose(man.sim_data["poa_direct"].mean(), 100.51798150298093)
    assert np.isclose(man.sim_data["poa_diffuse"].mean(), 68.24020664585301)
    assert np.isclose(man.sim_data["poa_sky_diffuse"].mean(), 67.04092283286448)
    assert np.isclose(man.sim_data["poa_ground_diffuse"].mean(), 1.1992838129885397)


def test_SolarWorkflowManager_configure_cec_module(
    pt_SolarWorkflowManager_poa: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa
    man.configure_cec_module(module="WINAICO WSx-240P6")
    assert isinstance(man.module, pd.Series)

    db = rk.solar.workflows.solar_workflow_manager.pvlib.pvsystem.retrieve_sam("CECMod")
    random_module = db.columns[3]
    man.configure_cec_module(module=random_module)
    assert isinstance(man.module, pd.Series)

    module = dict(
        BIPV="N",
        Date="12/14/2016",
        T_NOCT=45.7,
        A_c=1.673,
        N_s=60,
        I_sc_ref=10.82,
        V_oc_ref=42.8,
        I_mp_ref=10.01,
        V_mp_ref=37,
        alpha_sc=0.003246,
        beta_oc=-0.10272,
        a_ref=1.5532,
        I_L_ref=10.829,
        I_o_ref=1.12e-11,
        R_s=0.079,
        R_sh_ref=92.96,
        Adjust=14,
        gamma_r=-0.32,
        Version="NRELv1",
        PTC=347.2,
        Technology="Mono-c-Si",
    )
    man.configure_cec_module(module=module)
    assert isinstance(man.module, pd.Series)


@pytest.fixture
def pt_SolarWorkflowManager_cell_temp(
    pt_SolarWorkflowManager_poa: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa
    man.cell_temperature_from_sapm(mounting="glass_open_rack")

    return man


def test_SolarWorkflowManager_simulate_with_interpolated_single_diode_approximation(
    pt_SolarWorkflowManager_cell_temp: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_cell_temp
    man.simulate_with_interpolated_single_diode_approximation(
        module="WINAICO WSx-240P6",
    )

    print_testresults(man.sim_data["capacity_factor"])

    assert man.sim_data["capacity_factor"].shape == (54, 5)
    assert np.isclose(man.sim_data["capacity_factor"].mean(), 0.2363674873981133)
    assert np.isclose(man.sim_data["capacity_factor"].std(), 0.23436495032892843)
    assert np.isclose(man.sim_data["capacity_factor"].min(), 0.00013602136544332003)
    assert np.isclose(man.sim_data["capacity_factor"].max(), 0.8193065017891327)

    print(man.sim_data["module_dc_power_at_mpp"].mean())
    print(man.sim_data["module_dc_voltage_at_mpp"].mean())
    print(man.sim_data["total_system_generation"].mean())
    assert np.isclose(man.sim_data["module_dc_power_at_mpp"].mean(), 56.820853030607246)
    assert np.isclose(
        man.sim_data["module_dc_voltage_at_mpp"].mean(), 37.39738372317519
    )
    assert np.isclose(man.sim_data["total_system_generation"].mean(), 724.3133157683136)


@pytest.fixture
def pt_SolarWorkflowManager_sim(
    pt_SolarWorkflowManager_cell_temp: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_cell_temp
    man.simulate_with_interpolated_single_diode_approximation(
        module="WINAICO WSx-240P6",
    )

    return man


def test_SolarWorkflowManager_apply_inverter_losses(
    pt_SolarWorkflowManager_sim: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_sim
    man.placements["modules_per_string"] = 1
    man.placements["strings_per_inverter"] = 1
    del man.placements["capacity"]

    man.apply_inverter_losses(
        inverter="ABB__MICRO_0_25_I_OUTD_US_208__208V_", method="sandia"
    )

    print_testresults(man.sim_data["capacity_factor"])
    assert man.sim_data["capacity_factor"].shape == (54, 5)
    assert np.isclose(man.sim_data["capacity_factor"].mean(), 0.2233308735174432)
    assert np.isclose(man.sim_data["capacity_factor"].std(), 0.22766053886973034)
    assert np.isclose(man.sim_data["capacity_factor"].min(), -0.00031199041565443107)
    assert np.isclose(man.sim_data["capacity_factor"].max(), 0.7889831406846927)

    print(man.sim_data["total_system_generation"].mean())
    print(man.sim_data["inverter_ac_power_at_mpp"].mean())

    assert np.isclose(man.sim_data["total_system_generation"].mean(), 53.68695534660521)
    assert np.isclose(
        man.sim_data["inverter_ac_power_at_mpp"].mean(), 53.68695534660521
    )
