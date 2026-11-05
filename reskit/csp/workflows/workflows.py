import time
import warnings
from logging import warning
from os import PathLike

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.arraysetops import isin

from reskit import workflow_manager
from reskit.util.paths import is_path_like

from ... import weather as rk_weather
from .csp_workflow_manager import PTRWorkflowManager
from reskit.util.input_validation import WeatherInput, declare_inputs
from .dataset_handler import DatasetHandler
from reskit.util.provenance import record_provenance


@declare_inputs(
    PTRWorkflowManager,
    weather={
        "era5_path": WeatherInput(
            "ERA5",
            ("direct_horizontal_irradiance", "surface_wind_speed", "surface_air_temperature"),
            time_index_from="direct_horizontal_irradiance",
        )
    },
    files=("global_solar_atlas_dni_path", "global_solar_atlas_tamb_path", "elev_path"),
)
@record_provenance
def csp_ptr_era5(
    placements: pd.DataFrame,
    era5_path: str | PathLike | rk_weather.NCSource,
    global_solar_atlas_dni_path: str | float | np.ndarray,
    global_solar_atlas_tamb_path: str | PathLike | None = None,
    datasets: str | list[str] | None = None,
    cost_year: int = 2050,
    HTF_sel: list[str] = ["Heliosol", "SolarSalt", "Therminol"],
    elev_path: str | PathLike | list[float] | None = None,
    output_netcdf_path: str | None = None,
    output_variables: str | list[str] | None = None,
    return_self: bool | None = None,
    JITaccelerate: bool = False,
    verbose: bool = False,
    debug_vars: bool = False,
    onlynightuse: bool = True,
    fullvariation: bool = False,
    _validation: bool = False,
    time_slice: slice | None = None,
) -> PTRWorkflowManager | xr.Dataset | str:
    """
    This function is the overall workflow for the csp simulation and calls all subfunctions.
    It is a wrapper around the function "csp_ptr_era5_specific_dataset" below, to include the case of multiple datasets.
    Multiple datasets are refering to multiple heat transfer fluids and therefore also different power plants. In that case, for each placement the optimal HTF is selected.

    Parameters
    ----------
    placements: Pandas DataFrame
        Locations where to perform simulations at.
        Required columns: longitude, latitude, and at least one of the two area columns
        "land_area_m2" and "aperture_area_m2". If you give one column, the workflow computes
        the other one with the solar field density of the dataset. If you give both columns,
        the workflow keeps both values and does not use the solar field density.
        The columns "area" and "area_m2" are deprecated aliases of "land_area_m2". They give a
        DeprecationWarning, and RESKit 0.6.0 removes them.

    era5_path: str or rk_weather.NCSource
        Path to ERA5 weather data or an NCSource object.
        Required columns: "direct_horizontal_irradiance", "surface_wind_speed", "surface_air_temperature".

    global_solar_atlas_dni_path: str or float or np.ndarray
        Long-term average DNI data for placements. Could be provided as:
            - str: Path to raster file with DNI data from Global Solar Atlas.
            - float: Constant DNI value for all placements.
            - np.ndarray: Array of DNI values, shape must match (placements,) or (time, placements).

    global_solar_atlas_tamb_path: str, optional
        path to temperature data from global solar atlas. Used for selecting at which placement which HTF is optimal.

    datasetname: str, optional
        Name of the CSP technology dataset (e.g., "Heliosol", "SolarSalt").
        Defaults to "Validation 10". See csp/data/CSP_database.xlsx for options.

    cost_year: int, optional
        Year for cost calculation of the CSP plant.
        Defaults to 2050.

    HTF_sel: list of str, optional
        List of heat transfer fluids to consider for the simulation.
        Each heat transfer fluid corresponds to a different CSP technology dataset.
        Options: ["Heliosol", "SolarSalt", "Therminol"].

    elev_path: str or list, optional
        Elevation data for placements. Could be provided as:
            - str: Path to raster file with elevation data.
            - list: Elevation values for each placement.

    output_netcdf_path: str, optional
        Path to a file that you want to save your output NETCDF file at.
        Defaults to None.

    output_variables: list of {str, number}, optional
        Output variables of the simulation that you want to save, by their output names.
        If None, includes all suitable variables from placements, workflow parameters, simulation data, and time index.
        Interim variables which are not listed are dropped as soon as the workflow no longer needs them, which lowers its memory use.

    return_self: bool, optional
        Deprecated, removed in RESKit v1.0.0. If True, returns the workflow manager object
        instead of the results. By default the results are returned, like in the other
        workflows.

    JITaccelerate: bool, optional
        If True, enables JIT acceleration for some calculations.
        Defaults to False.

    verbose: bool, optional
        If True, prints progress information.
        Defaults to False.

    debug_vars: bool, optional
        If True, retains intermediate variables for debugging.
        Defaults to False.

    onlynightuse: bool, optional
        If True, optimizes plant size for night use only.
        Defaults to True.

    fullvariation: bool, optional
        If True, applies full variation in calculations.
        Defaults to False.

    _validation: bool, optional
        If True, runs in validation mode (some input data may be replaced).
        Defaults to False.

    time_slice : slice, optional
        Simulate only the time steps between time_slice.start and time_slice.stop, both
        inclusive, e.g. slice("2015-03-01", "2015-03-31 23:30"). Only these time steps
        are read from the weather source, which is strongly recommended for multi-year
        Zarr cloud stores. By default None, i.e. all time steps.

    Returns
    -------
    xarray.Dataset or str
        The results, or the path of the NetCDF file written. With several datasets, each
        placement is simulated with its optimal dataset, named in the variable 'dataset'.
        See csp_ptr_era5_specific_dataset for the variables.
    """
    _warn_return_self(return_self)

    # handle inputs for datasets
    single_dataset = False
    if datasets == None:
        # get datasets from HTF_sel and cost_year
        datasets = ["Dataset_" + htf + "_" + str(cost_year) for htf in HTF_sel]
        if len(datasets) == 1:
            single_dataset = True
            datasets = datasets[0]
    elif isinstance(datasets, str):
        # datasets=datasets
        single_dataset = True
    elif isinstance(datasets, list):
        # datasets=datasets
        if len(datasets) == 1:
            single_dataset = True
            datasets = datasets[0]
    else:
        raise TypeError(f"datasets got unknown datatype")

    if not single_dataset:
        assert is_path_like(global_solar_atlas_tamb_path)

    if single_dataset:  # only one dataset given
        with warnings.catch_warnings():
            # the warning about return_self came above already
            warnings.filterwarnings("ignore", message="The argument return_self", category=DeprecationWarning)
            output = csp_ptr_era5_specific_dataset(
                placements=placements,
                era5_path=era5_path,
                global_solar_atlas_dni_path=global_solar_atlas_dni_path,
                datasetname=datasets,
                elev_path=elev_path,
                output_netcdf_path=output_netcdf_path,
                output_variables=output_variables,
                return_self=return_self,
                JITaccelerate=JITaccelerate,
                verbose=verbose,
                debug_vars=debug_vars,
                onlynightuse=onlynightuse,
                fullvariation=fullvariation,
                _validation=_validation,
                time_slice=time_slice,
            )
        return output

    else:  # multiple datasets found
        # number the placements by position; a column location_id becomes the coordinate at the end
        placements = placements.reset_index(drop=True)
        location_ids = None
        if "location_id" in placements.columns:
            location_ids = placements["location_id"].to_numpy()
            placements = placements.drop(columns=["location_id"])

        # 1) split up placements for each htf (e.g. solar salt or heliosol)
        d = DatasetHandler(datasets)
        placements = d.split_placements(
            placements=placements,
            gsa_dni_path=global_solar_atlas_dni_path,
            gsa_tamb_path=global_solar_atlas_tamb_path,
        )
        del d

        # 2) run each simulation
        outputs = []
        for dataset in datasets:
            # select placements for current dataset
            placements_dataset = placements[placements["Dataset_opt"] == dataset]

            # skip empty batches
            if len(placements_dataset) == 0:
                continue

            # starting core simulation; the results are written once, after merging
            output_dataset = csp_ptr_era5_specific_dataset(
                placements=placements_dataset,
                era5_path=era5_path,
                global_solar_atlas_dni_path=global_solar_atlas_dni_path,
                datasetname=dataset,
                elev_path=elev_path,
                output_variables=output_variables,
                JITaccelerate=JITaccelerate,
                verbose=verbose,
                debug_vars=debug_vars,
                onlynightuse=onlynightuse,
                fullvariation=fullvariation,
                _validation=_validation,
                time_slice=time_slice,
            )

            # name the dataset of each placement, also if output_variables leaves it out
            output_dataset["dataset"] = ("location", np.full(len(placements_dataset), dataset))
            if "Dataset_opt" in output_dataset.variables:
                output_dataset = output_dataset.drop_vars("Dataset_opt")

            # the position in the given placements
            output_dataset["location"] = placements_dataset.index.to_numpy()

            # remember outputs for each dataset
            outputs.append(output_dataset)
            del output_dataset

        # 3) merge data
        output = xr.concat(outputs, dim="location").sortby("location")
        if location_ids is not None:
            output["location"] = location_ids

        if output_netcdf_path is None:
            return output
        # compress the time series, as WorkflowManager.to_xarray() does
        encoding = {name: dict(zlib=True) for name in output.data_vars if output[name].ndim > 1}
        output.to_netcdf(output_netcdf_path, encoding=encoding)
        return output_netcdf_path


@declare_inputs(
    PTRWorkflowManager,
    weather={
        "era5_path": WeatherInput(
            "ERA5",
            ("direct_horizontal_irradiance", "surface_wind_speed", "surface_air_temperature"),
            time_index_from="direct_horizontal_irradiance",
        )
    },
    files=("global_solar_atlas_dni_path", "elev_path"),
)
@record_provenance
def csp_ptr_era5_specific_dataset(
    placements: pd.DataFrame,
    era5_path: str | PathLike | rk_weather.NCSource,
    global_solar_atlas_dni_path: str | float | np.ndarray,
    datasetname: str = "Validation 10",
    elev_path: str | PathLike | list[float] | None = None,
    output_netcdf_path: str | None = None,
    output_variables: str | list[str] | None = None,
    return_self: bool | None = None,
    JITaccelerate: bool = False,
    verbose: bool = False,
    debug_vars: bool = False,
    onlynightuse: bool = True,
    fullvariation: bool = False,
    _validation: bool = False,
    time_slice: slice | None = None,
):
    """
    Calculates the heat output from the solar field based on parabolic trough technology (PTC).
    This workflow simulates the performance of a CSP plant for given placements and technology datasets.
    Not yet fully validated, but results are plausible.

    Date: 27.07.2021
    Author: David Franzmann, IEK-3

    Parameters
    ----------
    placements: Pandas DataFrame
        Locations where to perform simulations at.
        Required columns: longitude, latitude, and at least one of the two area columns
        "land_area_m2" and "aperture_area_m2". If you give one column, the workflow computes
        the other one with the solar field density of the dataset. If you give both columns,
        the workflow keeps both values and does not use the solar field density.
        The columns "area" and "area_m2" are deprecated aliases of "land_area_m2". They give a
        DeprecationWarning, and RESKit 0.6.0 removes them.

    era5_path: str or rk_weather.NCSource
        Path to ERA5 weather data or an NCSource object.
        Required columns: "direct_horizontal_irradiance", "surface_wind_speed", "surface_air_temperature".

    global_solar_atlas_dni_path: str or float or np.ndarray
        Long-term average DNI data for placements. Could be provided as:
            - str: Path to raster file with DNI data from Global Solar Atlas.
            - float: Constant DNI value for all placements.
            - np.ndarray: Array of DNI values, shape must match (placements,) or (time, placements).

    datasetname: str, optional
        Name of the CSP technology dataset (e.g., "Heliosol", "SolarSalt").
        Defaults to "Validation 10". See csp/data/CSP_database.xlsx for options.

    elev_path: str or list, optional
        Elevation data for placements. Could be provided as:
            - str: Path to raster file with elevation data.
            - list: Elevation values for each placement.

    output_netcdf_path: str, optional
        Path to a file that you want to save your output NETCDF file at.
        Defaults to None.

    output_variables: list of {str, number}, optional
        Output variables of the simulation that you want to save, by their output names.
        If None, includes all suitable variables from placements, workflow parameters, simulation data, and time index.
        Interim variables which are not listed are dropped as soon as the workflow no longer needs them, which lowers its memory use.

    return_self: bool, optional
        Deprecated, removed in RESKit v1.0.0. If True, returns the workflow manager object
        instead of the results. By default the results are returned, like in the other
        workflows.

    JITaccelerate: bool, optional
        If True, enables JIT acceleration for some calculations.
        Defaults to False.

    verbose: bool, optional
        If True, prints progress information.
        Defaults to False.

    debug_vars: bool, optional
        If True, retains intermediate variables for debugging.
        Defaults to False.

    onlynightuse: bool, optional
        If True, optimizes plant size for night use only.
        Defaults to True.

    fullvariation: bool, optional
        If True, applies full variation in calculations.
        Defaults to False.

    _validation: bool, optional
        If True, runs in validation mode (some input data may be replaced).
        Defaults to False.

    time_slice : slice, optional
        Simulate only the time steps between time_slice.start and time_slice.stop, both
        inclusive, e.g. slice("2015-03-01", "2015-03-31 23:30"). Only these time steps
        are read from the weather source, which is strongly recommended for multi-year
        Zarr cloud stores. By default None, i.e. all time steps.

    Returns
    -------
    xarray.Dataset or str
        The results, or the path of the NetCDF file written. The attribute 'units' of each
        variable gives its unit. Besides the placements columns, the variables are:

        Per placement:
        - dataset: the CSP technology dataset simulated
        - solar_multiple, storage_hours: the optimal size of the solar field relative to the
          power plant, and of the thermal storage in hours of power plant operation
        - solar_field_capacity, design_dni: the design thermal output of the solar field
          and the direct normal irradiance it is designed for
        - power_plant_capacity, storage_capacity: the electric capacity of the power plant
          and the thermal capacity of the storage
        - annual_net_electricity_output: the net electricity output over the simulated period
        - non_dispatchable_share: the share of the net output from heat which bypasses the
          storage; 0 with onlynightuse
        - lcoe: the levelized cost of electricity
        - lcoh_solar_field, capex_solar_field, totex_solar_field: the levelized cost of heat,
          the capital cost and the annual cost of the solar field alone
        - capex: the capital cost of the whole plant
        - annual_solar_field_heat, annual_parasitics_plant, annual_parasitics_solar_field:
          the heat delivered to the plant and the parasitic electricity demand over the
          simulated period

        Per time step and placement:
        - heat_to_htf, heat_to_plant: the heat absorbed by the heat transfer fluid, and the
          heat delivered to the plant after the heat losses
        - solar_field_heating: the heat needed to keep the heat transfer fluid above its
          minimum temperature
        - parasitics, parasitics_plant, parasitics_solar_field: the parasitic electricity
          demand in total, of the plant and of the solar field
        - solar_field_capacity_factor, solar_field_heating_capacity_factor: heat_to_plant
          and the negative solar_field_heating relative to solar_field_capacity
        - direct_normal_irradiance, solar_zenith

        Per day and placement:
        - daily_net_electricity_output, daily_non_dispatchable_electricity_output
        - daily_backup_heating_electricity: the electricity for solar_field_heating which
          the storage cannot cover
        - daily_parasitics: the parasitic electricity demand of the plant and the solar field
        - daily_capacity_factor: the daily net output relative to power_plant_capacity

        With return_self=True (deprecated), the workflow manager object instead.
    """
    _warn_return_self(return_self)

    # 1) Load input data
    wf = PTRWorkflowManager(placements)

    ptr_data = wf.loadPTRdata(
        datasetname=datasetname
    )  # PTRdata referes to the different csp models, i.e. helisol or solar salt: csp/data/CSP_database.xlsx
    wf.determine_area()  # either determines aperture_area from land_area, or the other way around

    # 3) read in Input data
    if verbose:
        tic_start = time.time()
        print(
            "Simulation started for {n_placements} placements.".format(n_placements=len(wf.placements)),
            flush=True,
        )
        print("Reading in Weather data.", flush=True)
    wf.read(
        variables=[  # "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            # "surface_pressure",
            "surface_air_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        time_index_from="direct_horizontal_irradiance",
        time_slice=time_slice,
        verbose=False,
    )

    wf.check_ERA5_input()

    if verbose:
        tic_read = time.time()
        print(
            "Data read in within {dt}s.".format(dt=str(tic_read - tic_start)),
            flush=True,
        )
        print("Starting preanalysis.", flush=True)
    # 4) get length of timesteps for later numpy sizing

    wf.get_timesteps()

    # apply elevation
    wf.apply_elevation(elev_path)
    wf.apply_azimuth()
    # 5) calculate the solar position based on pvlib

    wf.calculateSolarPosition()

    # calculate DNI from ERA5 to DNi convention
    # ERA5 DIN: Heat flux per horizontal plane
    # DNI convention: Heat flux per normal (to zenith) plane
    wf.direct_normal_irradiance_from_trigonometry()

    if _validation:
        # when doing the valitadion, the dni atlas was not finally processed,
        # so it was replaced with 1. As only the plant and not the weather simulation
        # was validated, this does not have an impact, as the same weather input
        # was used
        wf.placements["LRA_factor_direct_normal_irradiance"] = 1
    else:
        wf.adjust_variable_to_long_run_average(
            variable="direct_normal_irradiance",
            source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI_2020_03,
            real_long_run_average=global_solar_atlas_dni_path,
            real_lra_scaling=1000 / 24,  # cast to hourly average kWh
            nodata_fallback="source",
        )

    # manipulationof input values for variation calculation
    wf._applyVariation()  # only for developers, can be ignored otherwise

    # 6) doing selfmade calulations until Heat to HTF (Heat transfer fluid)
    wf.calculateIAM(a1=ptr_data["a1"], a2=ptr_data["a2"], a3=ptr_data["a3"])
    wf.calculateShadowLosses(method="wagner2011", SF_density=ptr_data["SF_density_direct"])
    wf.calculateWindspeedLosses(max_windspeed_threshold=ptr_data["maxWindspeed"])
    wf.calculateDegradationLosses(
        efficiencyDropPerYear=ptr_data["efficiencyDropPerYear"],
        lifetime=ptr_data["lifetime"],
    )
    wf.calculateHeattoHTF(
        eta_ptr_max=ptr_data["eta_ptr_max"],
        eta_cleaness=ptr_data["eta_cleaness"],
        eta_other=ptr_data["eta_other"],
    )

    wf.apply_capacity_sf()
    if not debug_vars:
        del (
            wf.sim_data["surface_wind_speed"],
            wf.sim_data["tracking_angle"],
            wf.sim_data["solar_altitude_angle_degree"],
            wf.sim_data["direct_horizontal_irradiance"],
        )
        del (
            wf.sim_data["eta_wind"],
            wf.sim_data["eta_degradation"],
            wf.sim_data["eta_shdw"],
        )

    if verbose:
        tic_pre = time.time()
        print("Preanalysis within {dt}s.".format(dt=str(tic_pre - tic_read)), flush=True)
        print("Starting core simulation of the solar field.", flush=True)
    # 7) calculation heat to plant with loss model
    wf.applyHTFHeatLossModel(
        calculationmethod="dersch2018",
        params={
            "b": ptr_data["b"],
            "relTMplant": ptr_data["relTMplant"],
            "maxHTFTemperature": ptr_data["maxHTFTemperature"],
            "JITaccelerate": JITaccelerate,  # TODO: from ptr manager
            "minHTFTemperature": ptr_data["minHTFTemperature"],
            "inletHTFTemperature": ptr_data["inletHTFTemperature"],
            "add_losses_coefficient": ptr_data["add_losses_coefficient"],
            "discretizationmethod": ptr_data["discretizationmethod"],
        },
    )

    if not debug_vars:
        del (
            wf.sim_data["surface_air_temperature"],
            wf.sim_data["HTF_mean_temperature_C"],
            wf.sim_data["Heat_Losses_W"],
            wf.sim_data["theta"],
            wf.sim_data["IAM"],
        )

    # 8) calculate Parasitic Losses of the plant
    wf.calculateParasitics(
        calculationmethod="dersch2018",  # 'gafurov2013',
        params={
            "PL_sf_fixed_W_per_m^2_ap": 1.486,
            "PL_sf_pumping_W_per_m^2_ap": 8.3,
            "PL_plant_fix": ptr_data["PL_plant_fix"],
            # 'PL_sf_track': ptr_data['PL_sf_track'], #gafurov
            # 'PL_sf_pumping': ptr_data['PL_sf_pumping'], #gafurov
            "PL_plant_pumping": ptr_data["PL_plant_pumping"],
            "PL_plant_other": ptr_data["PL_plant_other"],
        },
    )

    # 9) calculate economics
    wf.calculateEconomics_SolarField(
        WACC=ptr_data["WACC"],
        lifetime=ptr_data["lifetime"],
        calculationmethod="franzmann2021",
        params={
            "CAPEX_solar_field_EUR_per_m^2_aperture": ptr_data["CAPEX_solar_field_EUR_per_m^2_aperture"],
            "CAPEX_land_EUR_per_m^2_land": ptr_data["CAPEX_land_EUR_per_m^2_land"],
            "CAPEX_indirect_cost_perc_CAPEX": ptr_data["CAPEX_indirect_cost_perc_CAPEX"],
            "electricity_price_EUR_per_kWh": ptr_data["electricity_price_EUR_per_kWh"],
            "OPEX_perc_CAPEX": ptr_data["OPEX_perc_CAPEX"],
        },
    )

    if not return_self:
        # what plant sizing, electrical output, LCOE and capacity factors still read
        still_needed = [
            "HeattoPlant_W",
            "P_heating_W",
            "Parasitics_W_el",
            "solar_zenith_degree",
            "direct_normal_irradiance",
            "annuity",
        ]
        wf.release_sim_data(still_needed, wf.internal_names(output_variables))

    if verbose:
        tic_sf_sim = time.time()
        print(
            "Solar field simulation done in {dt}s.".format(dt=str(tic_sf_sim - tic_pre)),
            flush=True,
        )
        print("Starting optimizing plant electric output.", flush=True)

    wf.optimize_plant_size(onlynightuse=onlynightuse, fullvariation=fullvariation, debug_vars=debug_vars)

    # wf.optimize_heat_output_4D()
    # wf.calculateEconomics_Plant_Storage_4D()
    # wf.optimal_Plant_Configuration_4D()

    if verbose:
        tic_opt_plant = time.time()
        print(
            "Optimal Sizing done in  {dt}s.".format(dt=str(tic_opt_plant - tic_sf_sim)),
            flush=True,
        )

    wf.calculate_electrical_output(onlynightuse=onlynightuse, debug_vars=debug_vars)
    wf.calculate_LCOE()
    wf.calculateCapacityFactors()

    if verbose:
        tic_final = time.time()
        print(
            "Total simulation done in {dt}s.".format(dt=str(tic_final - tic_start)),
            flush=True,
        )

    if return_self:
        return wf
    else:
        return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables, release=True)


def _warn_return_self(return_self):
    """Warns that the argument return_self of the CSP workflows is deprecated."""
    if return_self is not None:
        warnings.warn(
            "The argument return_self of the CSP workflows is deprecated and will be removed in RESKit v1.0.0. "
            "The workflows return the results; to get the workflow manager, run its steps yourself.",
            DeprecationWarning,
            # past the caller in this module and the wrappers of record_provenance and declare_inputs
            stacklevel=5,
        )


##########################
# DEPRECATED NAMES (#226) #
##########################
# The names below were renamed for PEP 8 in RESKit v0.6.0. Each old name stays
# available as a warning wrapper until v1.0.0. Do not add new code here.


def CSP_PTR_ERA5(*args, **kwargs):
    """
    Deprecated alias of :func:`csp_ptr_era5`.

    Kept for backward compatibility and scheduled for removal in RESKit v1.0.0.
    Use :func:`csp_ptr_era5` instead. All arguments are passed through unchanged.
    """
    warnings.warn(
        "CSP_PTR_ERA5() is deprecated and will be removed in RESKit v1.0.0. Use csp_ptr_era5() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return csp_ptr_era5(*args, **kwargs)


def CSP_PTR_ERA5_specific_dataset(*args, **kwargs):
    """
    Deprecated alias of :func:`csp_ptr_era5_specific_dataset`.

    Kept for backward compatibility and scheduled for removal in RESKit v1.0.0.
    Use :func:`csp_ptr_era5_specific_dataset` instead. All arguments are passed through unchanged.
    """
    warnings.warn(
        "CSP_PTR_ERA5_specific_dataset() is deprecated and will be removed in RESKit v1.0.0. Use csp_ptr_era5_specific_dataset() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return csp_ptr_era5_specific_dataset(*args, **kwargs)
