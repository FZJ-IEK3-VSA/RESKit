import warnings

import geokit as gk
import numpy as np
import pandas as pd
import pytest

import reskit as rk
from reskit.solar import SolarWorkflowManager


def print_testresults(variable):
    print("mean: ", variable[0:140, :].mean())
    print("std: ", variable[0:140, :].std())
    print("min: ", variable[0:140, :].min())
    print("max: ", variable[0:140, :].max())


def _make_SolarWorkflowManager() -> SolarWorkflowManager:
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


def test_SolarWorkflowManager___init__():
    """Run _make_SolarWorkflowManager(), which other tests use as a factory."""
    _make_SolarWorkflowManager()


@pytest.fixture
def pt_SolarWorkflowManager_initialized() -> SolarWorkflowManager:
    return _make_SolarWorkflowManager()


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

    fallback_elev = -1000

    # first test None case without elev attribute in placements
    man.apply_elevation(elev=None, fallback_elev=fallback_elev)
    # must yield fallback value for all locations
    assert np.isclose(
        man.placements["elev"],
        [fallback_elev, fallback_elev, fallback_elev, fallback_elev, fallback_elev],
    ).all()

    # now test using the elevation from the placements dataframe
    base_elev = [90, 80, 70, 60, 50]
    man.placements["elev"] = base_elev
    man.apply_elevation(elev=None, fallback_elev=fallback_elev)
    # the elev data must not have been altered when None and 'elev' in attribute
    assert np.isclose(man.placements["elev"], base_elev).all()

    # then test scalar value
    man.apply_elevation(elev=120, fallback_elev=fallback_elev)
    # must yield this value for all locs
    assert np.isclose(man.placements["elev"], [120, 120, 120, 120, 120]).all()

    # next test iterable as new elev
    new_elev = [100, 120, 140, 160, 2000]
    man.apply_elevation(elev=new_elev, fallback_elev=fallback_elev)
    # must yield the same iterable
    assert np.isclose(man.placements["elev"], new_elev).all()

    # last test raster elevation, therefore redefine placements so that we also have a loc OUTSIDE the raster extent
    placements2 = pd.DataFrame()
    placements2["lon"] = [
        6.083,
        6.183,
        6.083,
        6.183,
        7.083,  # this is outside the CLC aachen clipped raster
    ]
    placements2["lat"] = [
        50.475,
        50.575,
        50.675,
        50.775,
        50.875,
    ]
    man2 = SolarWorkflowManager(placements2)

    man2.apply_elevation(
        elev=rk.TEST_DATA["clc-aachen_clipped.tif"], fallback_elev=fallback_elev
    )  # not an elevation file, but still a raster
    # must yield raster values, with fallback value for those placements outside the actual file coverage
    assert np.isclose(
        man2.placements["elev"],
        [
            2,
            36,
            18,
            18,
            fallback_elev,
        ],  # the last must be equal to fallback since outside raster
    ).all()

    # cover the case that raster clipped to extent is None since all placements are outside
    placements3 = pd.DataFrame()
    placements3["lon"] = [  # these are all outside the CLC aachen clipped raster
        16.083,
        16.183,
        16.083,
        16.183,
        16.083,
    ]
    placements3["lat"] = [
        50.475,
        50.575,
        50.675,
        50.775,
        50.875,
    ]
    man2 = SolarWorkflowManager(placements3)

    man2.apply_elevation(
        elev=rk.TEST_DATA["clc-aachen_clipped.tif"], fallback_elev=fallback_elev
    )  # not an elevation file, but still a raster
    # must yield raster values, with fallback value for those placements outside the actual file coverage
    assert np.isclose(
        man2.placements["elev"],
        [
            fallback_elev,
            fallback_elev,
            fallback_elev,
            fallback_elev,
            fallback_elev,
        ],  # the last must be equal to fallback since outside raster
    ).all()


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
    # (self):
    man = pt_SolarWorkflowManager_loaded

    man.determine_solar_position()

    assert man.sim_data["solar_azimuth"].shape == (140, 5)

    assert np.isclose(man.sim_data["solar_azimuth"].mean(), 181.73457432221727)
    assert np.isclose(man.sim_data["solar_azimuth"].std(), 90.1773419980912)
    assert np.isclose(man.sim_data["solar_azimuth"].min(), 23.08699931763124)
    assert np.isclose(man.sim_data["solar_azimuth"].max(), 355.8281473471921)

    assert np.isclose(man.sim_data["apparent_solar_zenith"].mean(), 108.92855989863853)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].std(), 26.93061253166759)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].min(), 72.96540182055152)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].max(), 152.5141565975026)


def test_SolarWorkflowManager_determine_solar_position_matches_spa_python(
    pt_SolarWorkflowManager_loaded: SolarWorkflowManager,
):
    # The vectorised SPA must give what pvlib's spa_python gives for each location on its own,
    # with that location's own pressure and temperature.
    import pvlib

    man = pt_SolarWorkflowManager_loaded
    man.determine_solar_position()

    for i, placement in enumerate(man.placements.itertuples()):
        expected = pvlib.solarposition.spa_python(
            man.time_index,
            latitude=placement.lat,
            longitude=placement.lon,
            altitude=placement.elev,
            pressure=man.sim_data["surface_pressure"][:, i],
            temperature=man.sim_data["surface_air_temperature"][:, i],
        )
        np.testing.assert_allclose(man.sim_data["solar_azimuth"][:, i], expected["azimuth"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            man.sim_data["apparent_solar_zenith"][:, i], expected["apparent_zenith"], rtol=0, atol=1e-9
        )


def test_SolarWorkflowManager_determine_solar_position_after_numba_spa(
    pt_SolarWorkflowManager_loaded: SolarWorkflowManager,
):
    # The CSP workflow's get_solarposition(method="nrel_numba") recompiles the shared pvlib.spa
    # module with numba, whose functions cannot broadcast; the solar position must still work.
    import pvlib

    with warnings.catch_warnings():  # pvlib announces the reload, unless spa is in numba mode already
        warnings.simplefilter("ignore")
        pvlib.solarposition.get_solarposition(
            pd.DatetimeIndex(["2020-06-21 12:00"], tz="UTC"), 50, 6, method="nrel_numba"
        )
    man = pt_SolarWorkflowManager_loaded
    man.determine_solar_position()

    assert not np.isnan(man.sim_data["solar_azimuth"]).any()


def test_SolarWorkflowManager_determine_solar_position_rounding_is_deprecated(
    pt_SolarWorkflowManager_loaded: SolarWorkflowManager,
):
    with pytest.deprecated_call():
        pt_SolarWorkflowManager_loaded.determine_solar_position(lon_rounding=1)


@pytest.fixture
def pt_SolarWorkflowManager_solpos(
    pt_SolarWorkflowManager_loaded: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded

    man.determine_solar_position()

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
    assert np.isclose(man.sim_data["solar_azimuth"].mean(), 177.8134358191862)
    assert np.isclose(man.sim_data["solar_azimuth"].std(), 34.89959323629754)
    assert np.isclose(man.sim_data["solar_azimuth"].min(), 124.69865719550728)
    assert np.isclose(man.sim_data["solar_azimuth"].max(), 231.17927128202086)

    assert np.isclose(man.sim_data["apparent_solar_zenith"].mean(), 80.64319960952488)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].std(), 6.1979174523484115)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].min(), 72.96540182055152)
    assert np.isclose(man.sim_data["apparent_solar_zenith"].max(), 91.88803175230056)


def test_SolarWorkflowManager_determine_extra_terrestrial_irradiance(
    pt_SolarWorkflowManager_solpos: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_solpos
    man.determine_extra_terrestrial_irradiance()

    print_testresults(man.sim_data["extra_terrestrial_irradiance"])

    assert man.sim_data["extra_terrestrial_irradiance"].shape == (140, 5)
    assert np.isclose(man.sim_data["extra_terrestrial_irradiance"].mean(), 1413.9980694079702)
    assert np.isclose(man.sim_data["extra_terrestrial_irradiance"].std(), 0.019625866056578487)
    assert np.isclose(man.sim_data["extra_terrestrial_irradiance"].min(), 1413.940576307916)
    assert np.isclose(man.sim_data["extra_terrestrial_irradiance"].max(), 1414.0192010311885)


def test_SolarWorkflowManager_determine_air_mass(
    pt_SolarWorkflowManager_solpos: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_solpos
    man.determine_air_mass(model="kastenyoung1989")

    print_testresults(man.sim_data["air_mass"])

    assert man.sim_data["air_mass"].shape == (140, 5)
    assert np.isclose(man.sim_data["air_mass"].mean(), 21.679498191479887)
    assert np.isclose(man.sim_data["air_mass"].std(), 10.851324252009809)
    assert np.isclose(man.sim_data["air_mass"].min(), 3.379357050766727)
    assert np.isclose(man.sim_data["air_mass"].max(), 29.0)


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
    assert np.isclose(man.sim_data["direct_normal_irradiance"].mean(), 166.82370763092024)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].std(), 201.48864938505164)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].min(), 0.0)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].max(), 717.8067691272702)


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
    assert np.isclose(man.sim_data["diffuse_horizontal_irradiance"].mean(), 48.7513428550373)
    assert np.isclose(man.sim_data["diffuse_horizontal_irradiance"].std(), 34.84584727571099)
    assert np.isclose(man.sim_data["diffuse_horizontal_irradiance"].min(), 0.15659047212134164)
    assert np.isclose(man.sim_data["diffuse_horizontal_irradiance"].max(), 125.27559193238976)


def test_SolarWorkflowManager_direct_normal_irradiance_from_trigonometry(
    pt_SolarWorkflowManager_loaded2: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded2

    man.direct_normal_irradiance_from_trigonometry()

    print_testresults(man.sim_data["direct_normal_irradiance"])

    assert man.sim_data["direct_normal_irradiance"].shape == (54, 5)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].mean(), 158.01422773462687)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].std(), 179.34864240250207)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].min(), 0.0)
    assert np.isclose(man.sim_data["direct_normal_irradiance"].max(), 615.5670198020749)


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
    assert np.isclose(man.sim_data["system_tilt"].std(), 14.579867708535026)
    assert np.isclose(man.sim_data["system_tilt"].min(), 20.0)
    assert np.isclose(man.sim_data["system_tilt"].max(), 74.41018307077114)

    assert np.isclose(man.sim_data["system_azimuth"].mean(), 185.82883513681966)
    assert np.isclose(man.sim_data["system_azimuth"].std(), 52.78589069423684)
    assert np.isclose(man.sim_data["system_azimuth"].min(), 99.71477147193693)
    assert np.isclose(man.sim_data["system_azimuth"].max(), 264.1714175691827)


def test_SolarWorkflowManager_determine_angle_of_incidence(
    pt_SolarWorkflowManager_all_irrad: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_all_irrad
    man.determine_angle_of_incidence()

    print_testresults(man.sim_data["angle_of_incidence"])

    assert man.sim_data["angle_of_incidence"].shape == (54, 5)
    assert np.isclose(man.sim_data["angle_of_incidence"].mean(), 56.57614971247222)
    assert np.isclose(man.sim_data["angle_of_incidence"].std(), 12.027451877050776)
    assert np.isclose(man.sim_data["angle_of_incidence"].min(), 33.43543174185137)
    assert np.isclose(man.sim_data["angle_of_incidence"].max(), 80.24856003048832)


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

    assert np.isclose(man.sim_data["poa_global"].mean(), 174.0033749286125)
    assert np.isclose(man.sim_data["poa_global"].std(), 173.28828557741616)
    assert np.isclose(man.sim_data["poa_global"].min(), 0.13328509297399485)
    assert np.isclose(man.sim_data["poa_global"].max(), 620.6729107955822)

    assert np.isclose(man.sim_data["poa_direct"].mean(), 102.65813932037659)
    assert np.isclose(man.sim_data["poa_diffuse"].mean(), 71.34523560823591)
    assert np.isclose(man.sim_data["poa_sky_diffuse"].mean(), 69.82323902496371)
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
    assert np.isclose(man.sim_data["cell_temperature"].mean(), 6.6976908109006255)
    assert np.isclose(man.sim_data["cell_temperature"].std(), 5.642518863938644)
    assert np.isclose(man.sim_data["cell_temperature"].min(), -3.2822952246943804)
    assert np.isclose(man.sim_data["cell_temperature"].max(), 21.165193061269655)

    # roof top PV should run hotter than open-field
    man.cell_temperature_from_sapm(mounting="glass_close_roof")

    print_testresults(man.sim_data["cell_temperature"])

    assert man.sim_data["cell_temperature"].shape == (54, 5)
    assert np.isclose(man.sim_data["cell_temperature"].mean(), 9.40129021468595)
    assert np.isclose(man.sim_data["cell_temperature"].std(), 8.249502141429812)
    assert np.isclose(man.sim_data["cell_temperature"].min(), -3.2472615808752097)
    assert np.isclose(man.sim_data["cell_temperature"].max(), 31.0702366955479)


def test_SolarWorkflowManager_apply_angle_of_incidence_losses_to_poa(
    pt_SolarWorkflowManager_poa: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa
    man.apply_angle_of_incidence_losses_to_poa()

    print_testresults(man.sim_data["poa_global"])
    assert man.sim_data["poa_global"].shape == (54, 5)
    assert np.isclose(man.sim_data["poa_global"].mean(), 168.64874042516342)
    assert np.isclose(man.sim_data["poa_global"].std(), 169.413388174897)
    assert np.isclose(man.sim_data["poa_global"].min(), 0.12759789566504143)
    assert np.isclose(man.sim_data["poa_global"].max(), 612.8618101192072)

    print(man.sim_data["poa_direct"].mean())
    print(man.sim_data["poa_diffuse"].mean())
    print(man.sim_data["poa_sky_diffuse"].mean())
    print(man.sim_data["poa_ground_diffuse"].mean())
    assert np.isclose(man.sim_data["poa_direct"].mean(), 100.43501459564703)
    assert np.isclose(man.sim_data["poa_diffuse"].mean(), 68.2137258295164)
    assert np.isclose(man.sim_data["poa_sky_diffuse"].mean(), 67.01444201652788)
    assert np.isclose(man.sim_data["poa_ground_diffuse"].mean(), 1.1992838129885397)


def test_SolarWorkflowManager_configure_cec_module(
    pt_SolarWorkflowManager_poa: SolarWorkflowManager,
) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa
    man.configure_cec_module(module="WINAICO WSx-240P6", tech_year=2050)
    assert isinstance(man.module, pd.Series)

    man.configure_cec_module(module="WINAICO WSx-240P6", tech_year=2030)
    assert isinstance(man.module, pd.Series)
    # check some sample values that must be adapted to 2030
    assert man.module["a_ref"] == 1.5783541935483871  # checked
    assert man.module["gamma_r"] == -0.4140129032258065  # checked

    db = rk.solar.workflows.solar_workflow_manager.pvlib.pvsystem.retrieve_sam("CECMod")
    random_module = db.columns[3]
    man.configure_cec_module(module=random_module, tech_year=None)
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
    man.configure_cec_module(module=module, tech_year=None)
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
    assert np.isclose(man.sim_data["capacity_factor"].mean(), 0.23621601709815493)
    assert np.isclose(man.sim_data["capacity_factor"].std(), 0.23416031861509165)
    assert np.isclose(man.sim_data["capacity_factor"].min(), 0.00013602136544332003)
    assert np.isclose(man.sim_data["capacity_factor"].max(), 0.8186140980163213)

    print(man.sim_data["module_dc_power_at_mpp"].mean())
    print(man.sim_data["module_dc_voltage_at_mpp"].mean())
    print(man.sim_data["total_system_generation"].mean())
    assert np.isclose(man.sim_data["module_dc_power_at_mpp"].mean(), 56.78444078225967)
    assert np.isclose(man.sim_data["module_dc_voltage_at_mpp"].mean(), 37.39337385334944)
    assert np.isclose(man.sim_data["total_system_generation"].mean(), 723.8148553423471)


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

    man.apply_inverter_losses(inverter="ABB__MICRO_0_25_I_OUTD_US_208__208V_", method="sandia")

    print_testresults(man.sim_data["capacity_factor"])
    assert man.sim_data["capacity_factor"].shape == (54, 5)
    assert np.isclose(man.sim_data["capacity_factor"].mean(), 0.2231838618104795)
    assert np.isclose(man.sim_data["capacity_factor"].std(), 0.22746190610551176)
    assert np.isclose(man.sim_data["capacity_factor"].min(), -0.00031199041565443107)
    assert np.isclose(man.sim_data["capacity_factor"].max(), 0.7883146154672291)

    print(man.sim_data["total_system_generation"].mean())
    print(man.sim_data["inverter_ac_power_at_mpp"].mean())

    assert np.isclose(man.sim_data["total_system_generation"].mean(), 53.65161490834479)
    assert np.isclose(man.sim_data["inverter_ac_power_at_mpp"].mean(), 53.65161490834479)


def test_SolarWorkflowManager_nan_values_tilt_azimuth_elev___init__() -> SolarWorkflowManager:
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
        None,
        30,
        35,
        40,
    ]
    placements["azimuth"] = [180, None, 180, 180, 180]
    placements["elev"] = [100, None, None, 100, 180]

    man = SolarWorkflowManager(placements)
    man.configure_cec_module(module="WINAICO WSx-240P6")

    # limit the input placements longitude to range of -180...180
    assert man.placements["lon"].between(-180, 180, inclusive="both").any()
    # limit the input placements latitude to range of -90...90
    assert man.placements["lat"].between(-90, 90, inclusive="both").any()
    # ensure the tracking parameter is correct

    # estimates tilt, azimuth and elev
    elev = 300  # fallback elevation
    man.estimate_missing_params(elev)

    assert ~man.placements["tilt"].isna().any()
    assert ~man.placements["azimuth"].isna().any()
    assert ~man.placements["elev"].isna().any()
