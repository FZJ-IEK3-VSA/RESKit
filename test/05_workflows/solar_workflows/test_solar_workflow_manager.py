import pandas as pd
import numpy as np
from reskit.solar import SolarWorkflowManager
import reskit as rk
import geokit as gk
import pytest


def print_testresults(variable):
    print('mean: ', variable[0:140,:].mean())
    print('std: ', variable[0:140,:].std())
    print('min: ', variable[0:140,:].min())
    print('max: ', variable[0:140,:].max())


def test_SolarWorkflowManager___init__() -> SolarWorkflowManager:
    # (self, placements):
    placements = pd.DataFrame()
    placements['lon'] = [6.083, 6.183, 6.083, 6.183, 6.083, ]
    placements['lat'] = [50.475, 50.575, 50.675, 50.775, 50.875, ]
    placements['capacity'] = [2000, 2500, 3000, 3500, 4000, ]
    placements['tilt'] = [20, 25, 30, 35, 40, ]
    placements['azimuth'] = [180, 180, 180, 180, 180]

    man = SolarWorkflowManager(placements)

    assert np.isclose(man.ext.xMin, 6.083000)
    assert np.isclose(man.ext.xMax, 6.183000)
    assert np.isclose(man.ext.yMin, 50.475000)
    assert np.isclose(man.ext.yMax, 50.875000)

    assert (man.placements['lon'] == placements['lon']).all()
    assert (man.placements['lat'] == placements['lat']).all()
    assert (man.placements['capacity'] == placements['capacity']).all()
    assert (man.placements['tilt'] == placements['tilt']).all()
    assert (man.placements['azimuth'] == placements['azimuth']).all()

    return man


@pytest.fixture
def pt_SolarWorkflowManager_initialized() -> SolarWorkflowManager:
    return test_SolarWorkflowManager___init__()


def test_SolarWorkflowManager_estimate_tilt_from_latitude(pt_SolarWorkflowManager_initialized):
    # (self, convention):
    man = pt_SolarWorkflowManager_initialized

    man.estimate_tilt_from_latitude("Ryberg2020")

    assert np.isclose(
        man.placements['tilt'],
        [39.0679049, 39.1082060, 39.1484058, 39.1885045, 39.2285025]
    ).all()

    man.estimate_tilt_from_latitude("(latitude-5)**2")

    assert np.isclose(
        man.placements['tilt'],
        [2067.975625, 2077.080625, 2086.205625, 2095.350625, 2104.515625]
    ).all()


def test_SolarWorkflowManager_estimate_azimuth_from_latitude(pt_SolarWorkflowManager_initialized):
    man = pt_SolarWorkflowManager_initialized

    man.estimate_azimuth_from_latitude()

    assert np.isclose(
        man.placements['azimuth'],
        [180, 180, 180, 180, 180]
    ).all()

    man.placements['lat'] *= -1
    man.locs = gk.LocationSet(man.placements[['lon', 'lat']].values)
    man.estimate_azimuth_from_latitude()

    assert np.isclose(
        man.placements['azimuth'],
        [0, 0, 0, 0, 0]
    ).all()


def test_SolarWorkflowManager_apply_elevation(pt_SolarWorkflowManager_initialized):
    man = pt_SolarWorkflowManager_initialized
    man.apply_elevation(120)

    assert np.isclose(
        man.placements['elev'],
        [120, 120, 120, 120, 120]
    ).all()

    man.apply_elevation(rk.TEST_DATA['gwa50-like.tif'])  # not an elevation file, but still a raster

    assert np.isclose(
        man.placements['elev'],
        [4.81529235, 4.54979848, 4.83163261, 5.10659551, 5.07869386]
    ).all()

    new_elev = [100, 120, 140, 160, 2000]
    man.apply_elevation(new_elev)

    assert np.isclose(
        man.placements['elev'],
        new_elev
    ).all()

    return man


@pytest.fixture
def pt_SolarWorkflowManager_loaded(pt_SolarWorkflowManager_initialized: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_initialized
    man.apply_elevation([100, 120, 140, 160, 2000])

    man.read(
        variables=["global_horizontal_irradiance",
                   "direct_horizontal_irradiance",
                   "surface_wind_speed",
                   "surface_pressure",
                   "surface_air_temperature",
                   "surface_dew_temperature", ],
        source_type="ERA5",
        source=rk.TEST_DATA['era5-like'],
        set_time_index=True,
        verbose=False
    )

    return man


def test_SolarWorkflowManager_determine_solar_position(pt_SolarWorkflowManager_loaded: SolarWorkflowManager) -> SolarWorkflowManager:
    # (self, lon_rounding=1, lat_rounding=1, elev_rounding=-2):
    man = pt_SolarWorkflowManager_loaded

    man.determine_solar_position(
        lon_rounding=1,
        lat_rounding=1,
        elev_rounding=-2,
    )

    assert man.sim_data['solar_azimuth'].shape == (140, 5)


    assert np.isclose(man.sim_data['solar_azimuth'].mean(), 181.75084452775852)
    assert np.isclose(man.sim_data['solar_azimuth'].std(), 90.18959294069582)
    assert np.isclose(man.sim_data['solar_azimuth'].min(), 23.100292572931437)
    assert np.isclose(man.sim_data['solar_azimuth'].max(), 355.8650905234781)

    assert np.isclose(man.sim_data['apparent_solar_zenith'].mean(), 108.93266465908583)
    assert np.isclose(man.sim_data['apparent_solar_zenith'].std(), 26.914599770957278)
    assert np.isclose(man.sim_data['apparent_solar_zenith'].min(), 72.98977919840057)
    assert np.isclose(man.sim_data['apparent_solar_zenith'].max(), 152.49005970814673)

    return man


@pytest.fixture
def pt_SolarWorkflowManager_solpos(pt_SolarWorkflowManager_loaded: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded

    man.determine_solar_position(
        lon_rounding=1,
        lat_rounding=1,
        elev_rounding=-2,
    )

    return man


def test_SolarWorkflowManager_filter_positive_solar_elevation(pt_SolarWorkflowManager_solpos: SolarWorkflowManager) -> SolarWorkflowManager:
    # (self):
    man = pt_SolarWorkflowManager_solpos

    man.filter_positive_solar_elevation()

    print_testresults(man.sim_data['solar_azimuth'])
    print_testresults(man.sim_data['apparent_solar_zenith'])

    assert man.sim_data['solar_azimuth'].shape == (54, 5)
    assert np.isclose(man.sim_data['solar_azimuth'].mean(), 177.8281611330465)
    assert np.isclose(man.sim_data['solar_azimuth'].std(), 34.898695009531934)
    assert np.isclose(man.sim_data['solar_azimuth'].min(), 124.71102348726265)
    assert np.isclose(man.sim_data['solar_azimuth'].max(), 231.1922838037285)

    assert np.isclose(man.sim_data['apparent_solar_zenith'].mean(), 80.66305060580946)
    assert np.isclose(man.sim_data['apparent_solar_zenith'].std(), 6.193300145511795)
    assert np.isclose(man.sim_data['apparent_solar_zenith'].min(), 72.98965034012224)
    assert np.isclose(man.sim_data['apparent_solar_zenith'].max(), 91.89378124249768)

    return man


def test_SolarWorkflowManager_determine_extra_terrestrial_irradiance(pt_SolarWorkflowManager_solpos: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_solpos
    man.determine_extra_terrestrial_irradiance()

    assert man.sim_data['extra_terrestrial_irradiance'].shape == (140, 5)
    assert np.isclose(man.sim_data['extra_terrestrial_irradiance'].mean(), 1413.998226484733)
    assert np.isclose(man.sim_data['extra_terrestrial_irradiance'].std(), 0.019249481046559633)
    assert np.isclose(man.sim_data['extra_terrestrial_irradiance'].min(), 1413.9625670547816)
    assert np.isclose(man.sim_data['extra_terrestrial_irradiance'].max(), 1414.0192010311885)

    return man


def test_SolarWorkflowManager_determine_air_mass(pt_SolarWorkflowManager_solpos: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_solpos
    man.determine_air_mass(model='kastenyoung1989')

    assert man.sim_data['air_mass'].shape == (140, 5)
    assert np.isclose(man.sim_data['air_mass'].mean(), 21.68991900861127)
    assert np.isclose(man.sim_data['air_mass'].std(), 10.849251271205912)
    assert np.isclose(man.sim_data['air_mass'].min(), 3.3839264244992067)
    assert np.isclose(man.sim_data['air_mass'].max(), 29.0)

    return man


@pytest.fixture
def pt_SolarWorkflowManager_loaded2(pt_SolarWorkflowManager_solpos: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_solpos
    man.filter_positive_solar_elevation()
    man.determine_extra_terrestrial_irradiance()
    man.determine_air_mass(model='kastenyoung1989')

    return man


def test_SolarWorkflowManager_apply_DIRINT_model(pt_SolarWorkflowManager_loaded2: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded2
    man.apply_DIRINT_model(use_pressure=True, use_dew_temperature=True)

    assert man.sim_data['direct_normal_irradiance'].shape == (54, 5)
    assert np.isclose(man.sim_data['direct_normal_irradiance'].mean(), 145.1130469731831)
    assert np.isclose(man.sim_data['direct_normal_irradiance'].std(), 229.4115609627865)
    assert np.isclose(man.sim_data['direct_normal_irradiance'].min(), 0.0)
    assert np.isclose(man.sim_data['direct_normal_irradiance'].max(), 706.2942463210451)


@pytest.fixture
def pt_SolarWorkflowManager_dni(pt_SolarWorkflowManager_loaded2: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded2
    man.apply_DIRINT_model(use_pressure=True, use_dew_temperature=True)

    return man


def test_SolarWorkflowManager_diffuse_horizontal_irradiance_from_trigonometry(pt_SolarWorkflowManager_dni: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_dni
    man.diffuse_horizontal_irradiance_from_trigonometry()

    assert man.sim_data['diffuse_horizontal_irradiance'].shape == (54, 5)
    assert np.isclose(man.sim_data['diffuse_horizontal_irradiance'].mean(), 55.12967978163645)
    assert np.isclose(man.sim_data['diffuse_horizontal_irradiance'].std(), 41.80376491895811)
    assert np.isclose(man.sim_data['diffuse_horizontal_irradiance'].min(), 0.0)
    assert np.isclose(man.sim_data['diffuse_horizontal_irradiance'].max(), 140.13512724209937)


def test_SolarWorkflowManager_direct_normal_irradiance_from_trigonometry(pt_SolarWorkflowManager_loaded2: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_loaded2

    man.direct_normal_irradiance_from_trigonometry()

    assert man.sim_data['direct_normal_irradiance'].shape == (54, 5)
    assert np.isclose(man.sim_data['direct_normal_irradiance'].mean(), 193.62480029771262)
    assert np.isclose(man.sim_data['direct_normal_irradiance'].std(), 252.973120077321)
    assert np.isclose(man.sim_data['direct_normal_irradiance'].min(), -0.0)
    assert np.isclose(man.sim_data['direct_normal_irradiance'].max(), 1309.323801737614)


@pytest.fixture
def pt_SolarWorkflowManager_all_irrad(pt_SolarWorkflowManager_dni: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_dni
    man.diffuse_horizontal_irradiance_from_trigonometry()

    return man


def test_SolarWorkflowManager_permit_single_axis_tracking(pt_SolarWorkflowManager_all_irrad: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_all_irrad
    man.permit_single_axis_tracking(
        max_angle=90,
        backtrack=True,
        gcr=0.2857142857142857,
    )

    assert man.sim_data['system_tilt'].shape == (54, 5)
    assert np.isclose(man.sim_data['system_tilt'].mean(), 46.36776761902535)
    assert np.isclose(man.sim_data['system_tilt'].std(), 14.57053047293401)
    assert np.isclose(man.sim_data['system_tilt'].min(), 20.0)
    assert np.isclose(man.sim_data['system_tilt'].max(), 74.30200035670491)

    assert np.isclose(man.sim_data['system_azimuth'].mean(), 185.84604997219606)
    assert np.isclose(man.sim_data['system_azimuth'].std(), 52.78824101665198)
    assert np.isclose(man.sim_data['system_azimuth'].min(), 99.71507906498667)
    assert np.isclose(man.sim_data['system_azimuth'].max(), 264.1287322591819)


def test_SolarWorkflowManager_determine_angle_of_incidence(pt_SolarWorkflowManager_all_irrad: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_all_irrad
    man.determine_angle_of_incidence()

    assert man.sim_data['angle_of_incidence'].shape == (54, 5)
    assert np.isclose(man.sim_data['angle_of_incidence'].mean(), 56.594491190253066)
    assert np.isclose(man.sim_data['angle_of_incidence'].std(), 12.023008726644628)
    assert np.isclose(man.sim_data['angle_of_incidence'].min(), 33.45866725976779)
    assert np.isclose(man.sim_data['angle_of_incidence'].max(), 80.25014934148591)


@pytest.fixture
def pt_SolarWorkflowManager_aoi(pt_SolarWorkflowManager_all_irrad: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_all_irrad
    man.determine_angle_of_incidence()

    return man


def test_SolarWorkflowManager_estimate_plane_of_array_irradiances(pt_SolarWorkflowManager_aoi: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_aoi
    man.estimate_plane_of_array_irradiances(
        transposition_model='perez',
        albedo=0.25
    )

    assert man.sim_data['poa_global'].shape == (54, 5)
    assert np.isclose(man.sim_data['poa_global'].mean(), 173.08216451910278)
    assert np.isclose(man.sim_data['poa_global'].std(), 184.95110192825993)
    assert np.isclose(man.sim_data['poa_global'].min(), 0.0)
    assert np.isclose(man.sim_data['poa_global'].max(), 631.7933238472692)

    assert np.isclose(man.sim_data['poa_direct'].mean(), 89.60771030069739)
    assert np.isclose(man.sim_data['poa_diffuse'].mean(), 83.47445421840538)
    assert np.isclose(man.sim_data['poa_sky_diffuse'].mean(), 81.97085406686897)
    assert np.isclose(man.sim_data['poa_ground_diffuse'].mean(), 1.5036001515364066)


@pytest.fixture
def pt_SolarWorkflowManager_poa(pt_SolarWorkflowManager_aoi: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_aoi
    man.estimate_plane_of_array_irradiances(
        transposition_model='perez',
        albedo=0.25
    )

    return man


def test_SolarWorkflowManager_cell_temperature_from_sapm(pt_SolarWorkflowManager_poa: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa

    man.cell_temperature_from_sapm(mounting='glass_open_rack')

    assert man.sim_data['cell_temperature'].shape == (54, 5)
    assert np.isclose(man.sim_data['cell_temperature'].mean(), 6.468636188762014)
    assert np.isclose(man.sim_data['cell_temperature'].std(), 6.21830546826928)
    assert np.isclose(man.sim_data['cell_temperature'].min(), -3.2801978934030758)
    assert np.isclose(man.sim_data['cell_temperature'].max(), 21.887346030975728)

    # roof top PV should run hotter than open-field
    man.cell_temperature_from_sapm(mounting='glass_close_roof')

    assert man.sim_data['cell_temperature'].shape == (54, 5)
    assert np.isclose(man.sim_data['cell_temperature'].mean(), 9.156048741650649)
    assert np.isclose(man.sim_data['cell_temperature'].std(), 9.005449074195452)
    assert np.isclose(man.sim_data['cell_temperature'].min(), -3.244002112200272)
    assert np.isclose(man.sim_data['cell_temperature'].max(), 31.957054506634336)


def test_SolarWorkflowManager_apply_angle_of_incidence_losses_to_poa(pt_SolarWorkflowManager_poa: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa
    man.apply_angle_of_incidence_losses_to_poa()

    assert man.sim_data['poa_global'].shape == (54, 5)
    assert np.isclose(man.sim_data['poa_global'].mean(), 167.04228278201626)
    assert np.isclose(man.sim_data['poa_global'].std(), 179.5561799423248)
    assert np.isclose(man.sim_data['poa_global'].min(), 0.0)
    assert np.isclose(man.sim_data['poa_global'].max(), 621.1083557294388)

    assert np.isclose(man.sim_data['poa_direct'].mean(), 87.1801383592781)
    assert np.isclose(man.sim_data['poa_diffuse'].mean(), 79.86214442273817)
    assert np.isclose(man.sim_data['poa_sky_diffuse'].mean(), 78.67731269029902)
    assert np.isclose(man.sim_data['poa_ground_diffuse'].mean(), 1.1848317324391617)


def test_SolarWorkflowManager_configure_cec_module(pt_SolarWorkflowManager_poa: SolarWorkflowManager) -> SolarWorkflowManager:
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
        Technology="Mono-c-Si")
    man.configure_cec_module(module=module)
    assert isinstance(man.module, pd.Series)


@pytest.fixture
def pt_SolarWorkflowManager_cell_temp(pt_SolarWorkflowManager_poa: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_poa
    man.cell_temperature_from_sapm(mounting='glass_open_rack')

    return man


def test_SolarWorkflowManager_simulate_with_interpolated_single_diode_approximation(pt_SolarWorkflowManager_cell_temp: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_cell_temp
    man.simulate_with_interpolated_single_diode_approximation(
        module='WINAICO WSx-240P6',
    )
    assert man.sim_data['capacity_factor'].shape == (54, 5)
    assert np.isclose(man.sim_data['capacity_factor'].mean(), 0.2344529922042244)
    assert np.isclose(man.sim_data['capacity_factor'].std(), 0.2492987459521858)
    assert np.isclose(man.sim_data['capacity_factor'].min(), 0.0)
    assert np.isclose(man.sim_data['capacity_factor'].max(), 0.8307926711175828)

    assert np.isclose(man.sim_data['module_dc_power_at_mpp'].mean(), 56.36062370195792)
    assert np.isclose(man.sim_data['module_dc_voltage_at_mpp'].mean(), 33.3797197789669)
    assert np.isclose(man.sim_data['total_system_generation'].mean(), 715.8269418538197)


@pytest.fixture
def pt_SolarWorkflowManager_sim(pt_SolarWorkflowManager_cell_temp: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_cell_temp
    man.simulate_with_interpolated_single_diode_approximation(
        module='WINAICO WSx-240P6',
    )

    return man


def test_SolarWorkflowManager_apply_inverter_losses(pt_SolarWorkflowManager_sim: SolarWorkflowManager) -> SolarWorkflowManager:
    man = pt_SolarWorkflowManager_sim
    man.placements['modules_per_string'] = 1
    man.placements['strings_per_inverter'] = 1
    del man.placements['capacity']

    man.apply_inverter_losses(
        inverter='ABB__MICRO_0_25_I_OUTD_US_208__208V_',
        method='sandia')

    assert man.sim_data['capacity_factor'].shape == (54, 5)
    assert np.isclose(man.sim_data['capacity_factor'].mean(), 0.22216578163156858)
    assert np.isclose(man.sim_data['capacity_factor'].std(), 0.24150846226066588)
    assert np.isclose(man.sim_data['capacity_factor'].min(), -0.00031199041565443107)
    assert np.isclose(man.sim_data['capacity_factor'].max(), 0.8000614268453871)

    assert np.isclose(man.sim_data['total_system_generation'].mean(), 53.40687657797603)
    assert np.isclose(man.sim_data['inverter_ac_power_at_mpp'].mean(), 53.40687657797603)
