import pandas as pd
import numpy as np
from reskit import (
    WorkflowQueue,
    WorkflowManager,
    distribute_workflow,
    execute_workflow_iteratively,
    TEST_DATA,
)
import reskit as rk
import geokit as gk
import xarray
import pytest
import osgeo


@pytest.fixture
def pt_wind_placements() -> pd.DataFrame:
    df = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])
    df["hub_height"] = np.linspace(100, 130, df.shape[0])
    df["capacity"] = 3000
    df["rotor_diam"] = 170
    df.loc[::2, "rotor_diam"] = 150
    return df


def test_WorkflowManager___init__():
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
    placements["hub_height"] = [
        140,
        140,
        140,
        140,
        140,
    ]
    placements["capacity"] = [2000, 3000, 4000, 5000, 6000]
    placements["rotor_diam"] = [136, 136, 136, 136, 136]

    man = WorkflowManager(placements)

    assert isinstance(man.placements, pd.DataFrame)
    assert "lon" in man.placements.columns
    assert "lat" in man.placements.columns
    assert not "geom" in man.placements.columns
    assert isinstance(man.locs, gk.LocationSet)

    assert isinstance(man.ext, gk.Extent)
    assert np.isclose(man.ext.xMin, 6.083)
    assert np.isclose(man.ext.xMax, 6.183)
    assert np.isclose(man.ext.yMin, 50.475)
    assert np.isclose(man.ext.yMax, 50.875)

    assert len(man.sim_data) == 0
    assert man.time_index is None
    assert len(man.workflow_parameters) == 0

    return man


@pytest.fixture
def pt_WorkflowManager_initialized() -> WorkflowManager:
    return test_WorkflowManager___init__()


def test_WorkflowManager_set_time_index(
    pt_WorkflowManager_initialized: WorkflowManager,
) -> WorkflowManager:
    man = pt_WorkflowManager_initialized
    man.set_time_index(
        pd.date_range("2020-01-01 00:00:00", "2020-02-01 00:00:00", freq="h")
    )
    assert man.time_index[10] == pd.Timestamp("2020-01-01 10:00:00")
    assert man._time_sel_ is None
    assert man._sim_shape_ == (len(man.time_index), 5)


def test_WorkflowManager_read(
    pt_WorkflowManager_initialized: WorkflowManager,
) -> WorkflowManager:
    man = pt_WorkflowManager_initialized
    man.read(
        variables=[
            "elevated_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=rk.TEST_DATA["era5-like"],
        set_time_index=True,
        verbose=False,
        spatial_interpolation_mode="bilinear",
        temporal_reindex_method="nearest",
    )

    assert man.sim_data["elevated_wind_speed"].shape == (140, 5)
    assert np.isclose(man.sim_data["elevated_wind_speed"].mean(), 7.770940192441002)
    assert np.isclose(man.sim_data["elevated_wind_speed"].std(), 2.821386193932622)
    assert np.isclose(man.sim_data["elevated_wind_speed"].min(), 0.6247099215674192)
    assert np.isclose(man.sim_data["elevated_wind_speed"].max(), 15.57823145237587)

    assert np.isclose(man.sim_data["surface_pressure"].mean(), 99177.21094376309)
    assert np.isclose(
        man.sim_data["surface_air_temperature"].mean(), 0.9192180687015453
    )


@pytest.fixture
def pt_WorkflowManager_loaded(
    pt_WorkflowManager_initialized: WorkflowManager,
) -> WorkflowManager:
    man = pt_WorkflowManager_initialized

    man.read(
        variables=[
            "elevated_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=rk.TEST_DATA["era5-like"],
        set_time_index=True,
        verbose=False,
        spatial_interpolation_mode="bilinear",
        temporal_reindex_method="nearest",
    )

    return man


def test_WorkflowManager_spatial_disagregation(
    pt_WorkflowManager_initialized: WorkflowManager,
) -> WorkflowManager:
    man = pt_WorkflowManager_initialized

    man.read(
        variables=[
            "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
        ],
        source_type="ERA5",
        source=rk.TEST_DATA["era5-like"],
        set_time_index=True,
        verbose=False,
        spatial_interpolation_mode="bilinear",
        temporal_reindex_method="nearest",
    )

    man.spatial_disaggregation(
        variable="global_horizontal_irradiance",
        source_high_resolution=TEST_DATA["gsa-ghi-like.tif"],
        source_low_resolution=rk.weather.GSAmeanSource.GHI_with_ERA5_pixel,
    )
    man.spatial_disaggregation(
        variable="direct_horizontal_irradiance",
        source_high_resolution=TEST_DATA["gsa-ghi-like.tif"],
        source_low_resolution=rk.weather.GSAmeanSource.GHI_with_ERA5_pixel,
    )


def test_WorkflowManager_adjust_variable_to_long_run_average(
    pt_WorkflowManager_loaded: WorkflowManager,
) -> WorkflowManager:
    man = pt_WorkflowManager_loaded
    man.adjust_variable_to_long_run_average(
        "elevated_wind_speed",
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=TEST_DATA["gwa100-like.tif"],
        real_lra_scaling=1,
        spatial_interpolation="linear-spline",
    )

    assert man.sim_data["elevated_wind_speed"].shape == (140, 5)
    assert np.isclose(man.sim_data["elevated_wind_speed"].mean(), 6.6490035160795395)
    assert np.isclose(man.sim_data["elevated_wind_speed"].std(), 2.4374198507417097)
    assert np.isclose(man.sim_data["elevated_wind_speed"].min(), 0.5486170217294893)
    assert np.isclose(man.sim_data["elevated_wind_speed"].max(), 13.853410433409616)


def test_WorkflowManager_adjust_variable_to_long_run_average_() -> WorkflowManager:
    # create a test placements dataframe
    columns = ["lat", "lon", "capacity"]
    data = [
        [50.475, 6.1, 100.1],  # middle
        [50.0085, 6.1, 100.1],  # corner
        [40, 6.1, 100.1],  # outside
    ]
    placements = pd.DataFrame(data, columns=columns)

    # make dummy wf instance
    wf = rk.solar.SolarWorkflowManager(placements)

    # test fallback to interpolation 'nearest'
    wf.sim_data["test_nearest"] = np.ones(shape=(1, placements.shape[0]))
    wf.adjust_variable_to_long_run_average(
        variable="test_nearest",
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=TEST_DATA["gsa-ghi-like.tif"],
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback=np.nan,
        spatial_interpolation="near",
    )
    assert np.isclose(wf.sim_data["test_nearest"][0][0], 0.9558724)  # checked
    assert np.isclose(wf.sim_data["test_nearest"][0][1], 0.97806027)  # checked
    assert np.isnan(wf.sim_data["test_nearest"][0][2])  # checked

    # test fallback to source data
    wf.sim_data["test_source"] = np.ones(shape=(1, placements.shape[0]))
    wf.adjust_variable_to_long_run_average(
        variable="test_source",
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=TEST_DATA["gsa-ghi-like.tif"],
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback=1.0,  # 1.0 means 1.0 x source data (no real_lra_scaling)
        spatial_interpolation="near",
    )
    assert np.isclose(wf.sim_data["test_nearest"][0][0], 0.9558724)  # checked
    assert np.isclose(wf.sim_data["test_nearest"][0][1], 0.97806027)  # checked
    assert np.isclose(
        wf.sim_data["test_source"][0][2], 1
    )  # checked, must be one since real_lra==source_lra, without scaling

    # TODO the following block must be removed 6/2024 once 'source' option is gone
    # test again with deprecated 'source' fallback
    wf.sim_data["test_source_deprecated"] = np.ones(shape=(1, placements.shape[0]))
    wf.adjust_variable_to_long_run_average(
        variable="test_source_deprecated",
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=TEST_DATA["gsa-ghi-like.tif"],
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback="source",  # deprecated, but must yield the same result
        spatial_interpolation="near",
    )
    # make sure the 'source' nodata_fallback yields the same as 1.0
    assert (
        wf.sim_data["test_source_deprecated"] == wf.sim_data["test_source"]
    ).all()  # checked

    # test fallback to callable
    def my_test_function(locs, source_long_run_average_value):
        """Some random function to generate a lat/lon and source dependent value"""
        assert [
            isinstance(loc, osgeo.ogr.Geometry) for loc in locs
        ]  # just make sure 'locs' is what we expect
        return source_long_run_average_value * 2  # return 2 x source value

    wf.sim_data["test_callable"] = np.ones(shape=(1, placements.shape[0]))
    wf.adjust_variable_to_long_run_average(
        variable="test_callable",
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=TEST_DATA["gsa-ghi-like.tif"],
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback=my_test_function,  # should yield 2 x source data (no real_lra_scaling)
        spatial_interpolation="near",
    )
    assert np.isclose(wf.sim_data["test_callable"][0][0], 0.9558724)  # checked
    assert np.isclose(wf.sim_data["test_callable"][0][1], 0.97806027)  # checked
    assert np.isclose(
        wf.sim_data["test_callable"][0][2], 2
    )  # checked, my_test_function yields 2x source data, hence factor 2

    # now test fallback to another raster with a different coordinate set
    # define coordinates within and outside of the Aachen clipped CLC raster
    columns = ["lat", "lon", "capacity"]
    data2 = [
        [50.475, 6.1, 100.1],  # inside
        [50.2, 6.1, 100.1],  # outside source
        [40, 6.1, 100.1],  # outside fallback raster
    ]
    placements2 = pd.DataFrame(data2, columns=columns)
    wf2 = rk.solar.SolarWorkflowManager(placements2)
    # abuse the (slightly smaller) clc raster as main and the gsa-ghi-like raster (with order of magnitude 10x smaller) as fallback
    # the last point must still be outside of the fallback raster and will hence be nan
    wf2.sim_data["test_raster"] = np.ones(shape=(1, placements.shape[0]))
    wf2.adjust_variable_to_long_run_average(
        variable="test_raster",
        source_long_run_average=rk.weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=TEST_DATA["clc-aachen_clipped.tif"],
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
        nodata_fallback=TEST_DATA["gsa-ghi-like.tif"],
        spatial_interpolation="near",
    )
    assert np.isclose(wf2.sim_data["test_raster"][0][0], 8.04380701)
    assert np.isclose(
        wf2.sim_data["test_raster"][0][1], 0.02358450101346743
    )  # no correction applied to nodata_fallback raster values, hence the above x 24/1000
    assert np.isnan(wf2.sim_data["test_raster"][0][2])


def test_WorkflowManager_apply_loss_factor(
    pt_WorkflowManager_loaded: WorkflowManager,
) -> WorkflowManager:
    man = pt_WorkflowManager_loaded
    man.sim_data["capacity_factor"] = man.sim_data["elevated_wind_speed"].copy()
    man.apply_loss_factor(0.05, variables=["capacity_factor"])

    assert man.sim_data["capacity_factor"].shape == (140, 5)
    assert np.isclose(man.sim_data["capacity_factor"].mean(), 7.382393182818952)
    assert np.isclose(man.sim_data["capacity_factor"].std(), 2.680316884235991)
    assert np.isclose(man.sim_data["capacity_factor"].min(), 0.5934744254890483)
    assert np.isclose(man.sim_data["capacity_factor"].max(), 14.799319879757077)

    man.sim_data["capacity_factor_2"] = man.sim_data["elevated_wind_speed"].copy()
    man.apply_loss_factor(lambda x: 0.05, variables=["capacity_factor_2"])

    assert man.sim_data["capacity_factor_2"].shape == (140, 5)
    assert np.isclose(man.sim_data["capacity_factor_2"].mean(), 7.382393182818952)
    assert np.isclose(man.sim_data["capacity_factor_2"].std(), 2.680316884235991)
    assert np.isclose(man.sim_data["capacity_factor_2"].min(), 0.5934744254890483)
    assert np.isclose(man.sim_data["capacity_factor_2"].max(), 14.799319879757077)


def test_WorkflowManager_register_workflow_parameter(
    pt_WorkflowManager_loaded: WorkflowManager,
) -> WorkflowManager:
    man = pt_WorkflowManager_loaded
    man.register_workflow_parameter(key="hats", value=12)

    assert "hats" in man.workflow_parameters
    assert man.workflow_parameters["hats"] == 12


def test_WorkflowManager_to_xarray(
    pt_WorkflowManager_loaded: WorkflowManager,
) -> WorkflowManager:
    man = pt_WorkflowManager_loaded

    xds = man.to_xarray(
        output_netcdf_path=None,
        output_variables=None,
        _intermediate_dict=False,
    )

    assert xds["location"].shape == (5,)
    assert xds["lon"].shape == (5,)
    assert xds["lat"].shape == (5,)
    assert xds["hub_height"].shape == (5,)
    assert xds["capacity"].shape == (5,)
    assert xds["rotor_diam"].shape == (5,)
    assert xds["time"].shape == (140,)
    assert xds["elevated_wind_speed"].shape == (140, 5)
    assert xds["surface_pressure"].shape == (140, 5)
    assert xds["surface_air_temperature"].shape == (140, 5)

    assert np.isclose(float(xds["location"].fillna(0).mean()), 2.0)
    assert np.isclose(float(xds["lon"].fillna(0).mean()), 6.123)
    assert np.isclose(float(xds["lat"].fillna(0).mean()), 50.675000000000004)
    assert np.isclose(float(xds["hub_height"].fillna(0).mean()), 140.0)
    assert np.isclose(float(xds["capacity"].fillna(0).mean()), 4000.0)
    assert np.isclose(float(xds["rotor_diam"].fillna(0).mean()), 136.0)
    assert np.isclose(
        float(xds["elevated_wind_speed"].fillna(0).mean()), 7.770940192441002
    )
    assert np.isclose(
        float(xds["surface_pressure"].fillna(0).mean()), 99177.21094376309
    )
    assert np.isclose(
        float(xds["surface_air_temperature"].fillna(0).mean()), 0.9192180687015453
    )

    ##
    xds = man.to_xarray(
        output_netcdf_path=None,
        output_variables=[
            "lon",
            "lat",
            "elevated_wind_speed",
            "surface_air_temperature",
        ],
        _intermediate_dict=False,
    )

    assert "lon" in xds.variables
    assert "lat" in xds.variables
    assert "elevated_wind_speed" in xds.variables
    assert "surface_air_temperature" in xds.variables
    assert not "hub_height" in xds.variables
    assert not "capacity" in xds.variables
    assert not "rotor_diam" in xds.variables
    assert not "surface_pressure" in xds.variables


def simple_workflow(placements, era5_path, var1, var2):
    man = WorkflowManager(placements)

    man.read(
        variables=[
            "elevated_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
        spatial_interpolation_mode="bilinear",
        temporal_reindex_method="nearest",
    )

    man.sim_data["capacity_factor"] = man.sim_data["elevated_wind_speed"] * var1
    man.register_workflow_parameter("var2", var2)

    return man.to_xarray()


def test_distribute_workflow():
    xds = distribute_workflow(
        workflow_function=simple_workflow,
        placements=pd.read_csv(TEST_DATA["turbine_placements.csv"]),
        era5_path=TEST_DATA["era5-like"],
        var1=0.5,
        var2="pants",
        jobs=2,
        max_batch_size=None,
        intermediate_output_dir=None,
    )

    assert xds["time"].shape == (140,)
    assert xds["location"].shape == (560,)
    assert xds["capacity"].shape == (560,)
    assert xds["hub_height"].shape == (560,)
    assert xds["rotor_diam"].shape == (560,)
    assert xds["lon"].shape == (560,)
    assert xds["lat"].shape == (560,)
    assert xds["elevated_wind_speed"].shape == (140, 560)
    assert xds["surface_pressure"].shape == (140, 560)
    assert xds["surface_air_temperature"].shape == (140, 560)
    assert xds["capacity_factor"].shape == (140, 560)

    assert np.isclose(float(xds["location"].fillna(0).mean()), 279.5)
    assert np.isclose(float(xds["capacity"].fillna(0).mean()), 4000.0)
    assert np.isclose(float(xds["hub_height"].fillna(0).mean()), 120.0)
    assert np.isclose(float(xds["rotor_diam"].fillna(0).mean()), 140.0)
    assert np.isclose(float(xds["lon"].fillna(0).mean()), 6.16945196229404)
    assert np.isclose(float(xds["lat"].fillna(0).mean()), 50.80320853112445)
    assert np.isclose(
        float(xds["elevated_wind_speed"].fillna(0).mean()), 7.734400146016037
    )
    assert np.isclose(
        float(xds["surface_pressure"].fillna(0).mean()), 100262.81729624895
    )
    assert np.isclose(
        float(xds["surface_air_temperature"].fillna(0).mean()), 1.7100568364611306
    )
    assert np.isclose(
        float(xds["capacity_factor"].fillna(0).mean()), 3.8672000730080187
    )


def test_WorkflowQueue():
    # Create a queue
    queue = WorkflowQueue(
        workflow=simple_workflow,
        era5_path=TEST_DATA["era5-like"],
    )

    # append jobs to queue
    placements = pd.read_csv(TEST_DATA["turbine_placements.csv"])

    queue.append(
        key="run_1",
        placements=placements.iloc[::2, :],
        var1=0.5,
        var2="pants",
    )

    queue.append(
        key="run_2",
        placements=placements.iloc[1::3, :],
        var1=0.75,
        var2="cats",
    )

    queue.append(
        key="run_3",
        placements=placements.iloc[::4, :],
        var1=0.5,
        var2="rectangle",
    )

    queue.append(
        key="run_4",
        placements=placements.iloc[1::5, :],
        var1=0.75,
        var2="plane",
    )

    # Submit jobs
    results = queue.execute(jobs=2)

    # Done!
    assert results["run_1"]["capacity_factor"].shape == (140, 280)
    assert results["run_2"]["capacity_factor"].shape == (140, 187)
    assert results["run_3"]["capacity_factor"].shape == (140, 140)
    assert results["run_4"]["capacity_factor"].shape == (140, 112)

    assert np.isclose(
        float(results["run_1"]["capacity_factor"].mean()), 3.8682375177743236
    )
    assert np.isclose(
        float(results["run_2"]["capacity_factor"].mean()), 5.803744731787748
    )
    assert np.isclose(
        float(results["run_3"]["capacity_factor"].mean()), 3.8700743523128716
    )
    assert np.isclose(
        float(results["run_4"]["capacity_factor"].mean()), 5.8054026403965135
    )

    assert results["run_1"].var2 == "pants"
    assert results["run_2"].var2 == "cats"
    assert results["run_3"].var2 == "rectangle"
    assert results["run_4"].var2 == "plane"


def test_execute_workflow_iteratively(pt_wind_placements):
    gen = execute_workflow_iteratively(
        workflow=rk.wind.onshore_wind_merra_ryberg2019_europe,
        weather_path_varname="merra_path",
        zoom=None,
        # workflow_args:
        placements=pt_wind_placements,
        merra_path=TEST_DATA["merra-like"],
        gwa_50m_path=TEST_DATA["gwa50-like.tif"],
        clc2012_path=TEST_DATA["clc-aachen_clipped.tif"],
    )

    assert gen.roughness.shape == (560,)
    assert np.isclose(gen.roughness.mean(), 0.21235714)
    assert np.isclose(gen.roughness.min(), 0.005)
    assert np.isclose(gen.roughness.max(), 1.2)
    assert np.isclose(gen.roughness.std(), 0.22275803)

    assert gen.elevated_wind_speed.shape == (71, 560)
    assert np.isclose(gen.elevated_wind_speed.mean(), 7.25949888)
    assert np.isclose(gen.elevated_wind_speed.min(), 1.65055839)
    assert np.isclose(gen.elevated_wind_speed.max(), 14.22762999)
    assert np.isclose(gen.elevated_wind_speed.std(), 2.48011369)

    assert gen.capacity_factor.shape == (71, 560)
    assert np.isclose(gen.capacity_factor.mean(), 0.57601636)
    assert np.isclose(gen.capacity_factor.min(), 0.0)
    assert np.isclose(gen.capacity_factor.max(), 0.99326205)
    assert np.isclose(gen.capacity_factor.std(), 0.3504081)
