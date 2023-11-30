import reskit as rk
import pandas as pd
import numpy as np
from reskit import TEST_DATA
import pytest

#%%
# @pytest.fixture
def pt_pv_placements() -> pd.DataFrame:
    placements = pd.DataFrame()
    # noor 2 ptr plant, morocco
    placements["lon"] = [-6.8, -6.8, -6.8]  # Longitude  # [ -6.8, -6.8, -6.8]
    placements["lat"] = [
        31.0,
        31.4,
        31.0,
    ]  # Latitude  # [ 31.0, 31.4, 31.0,]
    placements["area_m2"] = [1e6, 5e6, 6e6]
    repeats = int(3 / 3)
    placements = placements.loc[placements.index.repeat(repeats)].reset_index(drop=True)
    return placements


# Make a placements dataframe

#%%
@pytest.mark.skip(
    reason="Not working on calamari. Tested locally at 01.11.2022/d.franzmann@fz-juelich.de"
)
def test_CSP_PTR_ERA5(pt_pv_placements):

    # local
    era5_path = r"R:\data\gears\weather\ERA5\processed\4\7\6\2015"
    global_solar_atlas_dni_path = r"R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF\DNI.tif"
    global_solar_atlas_tamb_path = r"R:\data\gears\geography\irradiance\global_solar_atlas_v2.5\World_TEMP_GISdata_LTAy_GlobalSolarAtlas_GEOTIFF\TEMP.tif"

    # cluster
    era5_path = r"/storage/internal/data/gears/weather/ERA5/processed/4/7/6/2015/"
    global_solar_atlas_dni_path = r"/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif"
    global_solar_atlas_tamb_path = r"/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_TEMP_GISdata_LTAy_GlobalSolarAtlas_GEOTIFF/TEMP.tif"

    out = rk.csp.CSP_PTR_ERA5(
        placements=pt_pv_placements,
        era5_path=era5_path,
        global_solar_atlas_dni_path=global_solar_atlas_dni_path,
        global_solar_atlas_tamb_path=global_solar_atlas_tamb_path,
        verbose=True,
        cost_year=2030,
        JITaccelerate=False,
        return_self=False,
        debug_vars=True,
        onlynightuse=True,
    )

    print("Simulation done")

    # datasets
    a = np.array(
        ["Dataset_SolarSalt_2030", "Dataset_Therminol_2030", "Dataset_SolarSalt_2030"]
    )
    assert (out["datasetname"].values == a).all()

    assert np.allclose(
        out["LRA_factor_direct_normal_irradiance"].values,
        [0.92240995, 0.85188336, 0.92240995],
    )

    # direct_horizontal_irradiance:
    assert out["direct_horizontal_irradiance"].values.shape == (8760, 3)
    assert np.isclose(
        out["direct_horizontal_irradiance"].values.mean(), 189.95024229234605
    )
    assert np.isclose(
        out["direct_horizontal_irradiance"].values.std(), 268.22838885782073
    )
    assert np.isclose(out["direct_horizontal_irradiance"].values.min(), 0.0)
    assert np.isclose(
        out["direct_horizontal_irradiance"].values.max(), 966.579790643025
    )

    # direct_horizontal_irradiance:
    assert out["direct_normal_irradiance"].values.shape == (8760, 3)
    assert np.isclose(out["direct_normal_irradiance"].values.mean(), 278.73085651103776)
    assert np.isclose(out["direct_normal_irradiance"].values.std(), 332.3526428406074)
    assert np.isclose(out["direct_normal_irradiance"].values.min(), 0.0)
    assert np.isclose(out["direct_normal_irradiance"].values.max(), 982.451222209881)

    # HeattoHTF_W
    assert out["HeattoHTF_W"].values.shape == (8760, 3)
    assert np.isclose(out["HeattoHTF_W"].values.mean(), 209780548.79719985)
    assert np.isclose(out["HeattoHTF_W"].values.std(), 324059983.1510693)
    assert np.isclose(out["HeattoHTF_W"].values.min(), 0.0)
    assert np.isclose(out["HeattoHTF_W"].values.max(), 1245394974.4195464)

    # HeattoPlant_W
    assert out["HeattoPlant_W"].values.shape == (8760, 3)
    assert np.isclose(out["HeattoPlant_W"].values.mean(), 159338864.09594935)
    assert np.isclose(out["HeattoPlant_W"].values.std(), 276057742.9582942)
    assert np.isclose(out["HeattoPlant_W"].values.min(), 0.0)
    assert np.isclose(out["HeattoPlant_W"].values.max(), 1102811547.598732)

    # P_heating_W
    assert out["P_heating_W"].values.shape == (8760, 3)
    assert np.isclose(out["P_heating_W"].values.mean(), 8567187.817746798)
    assert np.isclose(out["P_heating_W"].values.std(), 20261718.074392248)
    assert np.isclose(out["P_heating_W"].values.min(), 0.0)
    assert np.isclose(out["P_heating_W"].values.max(), 70676641.74470554)

    # sm_opt
    a = np.array([2.0, 2.0, 2.0])
    assert np.isclose(out["sm_opt"].values, a).all()
    # tes_opt
    a = np.array([12, 12, 12])
    assert np.isclose(out["tes_opt"].values, a).all()

    # Power_net_total_per_day_Wh
    assert out["Power_net_total_per_day_Wh"].values.shape == (365, 3)
    assert np.isclose(
        out["Power_net_total_per_day_Wh"].values.mean(), 1082442335.686746
    )
    assert np.isclose(out["Power_net_total_per_day_Wh"].values.std(), 797437011.9432147)
    assert np.isclose(out["Power_net_total_per_day_Wh"].values.min(), 0.0)
    assert np.isclose(out["Power_net_total_per_day_Wh"].values.max(), 2560268088.640056)

    # P_backup_heating_daily_Wh_el
    assert out["P_backup_heating_daily_Wh_el"].values.shape == (365, 3)
    assert np.isclose(
        out["P_backup_heating_daily_Wh_el"].values.mean(), 6517471.755673256
    )
    assert np.isclose(
        out["P_backup_heating_daily_Wh_el"].values.std(), 62620895.67712642
    )
    assert np.isclose(out["P_backup_heating_daily_Wh_el"].values.min(), 0.0)
    assert np.isclose(
        out["P_backup_heating_daily_Wh_el"].values.max(), 1054026635.4118232
    )

    # lcoe_EURct_per_kWh_el
    a = np.array([14.875259498468504, 17.115158016447037, 14.875259498468509])
    assert np.isclose(out["lcoe_EURct_per_kWh_el"].values, a).all()


if __name__ == "__main__":

    placements = pt_pv_placements()
    test_CSP_PTR_ERA5(placements)
