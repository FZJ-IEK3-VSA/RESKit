import json
import os
import time
from importlib import import_module

import numpy as np
import xarray as xr


def _import_data_extraction_tool():
    """Import ParFlow data extraction tool from vendored module."""
    try:
        import sys
        from pathlib import Path

        vendors_path = Path(__file__).parent.parent / "_external_module" / "parflow_600m_runs"
        if str(vendors_path) not in sys.path:
            sys.path.insert(0, str(vendors_path))

        return import_module("parflow_data_extraction")
    except Exception as exc:
        raise ImportError(
            "ParFlow data extraction dependency is not available. "
            "Ensure parflow_data_extraction.py is present in 'reskit/hydro/_external_module/'. "
            "See https://github.com/HPSCTerrSys/ParFlow_DataRetrievalAndExtraction"
        ) from exc


def retrieve_discharge_data(
    year,
    latitudes,
    longitudes,
    root_dir,
    include_neighbours=False,
    ctrl_file_path=None,
    max_retries=10,
    retry_delay_seconds=5,
):
    """Extract discharge time series from ParFlow datasource for given coordinates."""
    data_extraction_tool = _import_data_extraction_tool()

    if ctrl_file_path is None:
        ctrl_file_path = os.path.join(root_dir, "locations.json")

    # data_url = (
    #     "https://service.tereno.net/thredds/dodsC/forecastnrw/products/tmp_uclouvain/"
    #     f"sfd_DE05_ECMWF-HRES_hindcast_r1i1p2_FZJ-IBG3-ParFlowCLM380_hgfadapter-h00-v03bJuwelsGpuProdClimatologyTl_1day_{year}0101-{year}1231.nc"
    # )

    ### updated URL
    data_url = (
        "https://service.tereno.net/thredds/dodsC/forecastnrw/products/ParFlow-DE06-HC_v03/"
        f"sfd_DE05_ECMWF-HRES_hindcast_r1i1p2_FZJ-IBG3-ParFlowCLM380_hgfadapter-h00-v03bJuwelsGpuProdClimatologyTl_1day_{year}0101-{year}1231.nc"
    )

    ctrl_file = {
        "indicatorFile": os.path.join(
            root_dir,
            "DE-0055_INDICATOR_regridded_rescaled_SoilGrids250-v2017_BGRvector_newAllv.nc",
        ),
        "locations": [],
    }

    for i in range(len(latitudes)):
        ctrl_file["locations"].append(
            {
                "locationID": f"location_{i}",
                "locationLat": float(latitudes[i]),
                "locationLon": float(longitudes[i]),
                "simData": data_url,
                "depth": 0.2,
            }
        )

    os.makedirs(os.path.dirname(ctrl_file_path), exist_ok=True)
    with open(ctrl_file_path, "w") as f:
        json.dump(ctrl_file, f)

    # The vendored helper accepts only (runctrl_file, output_format).
    # Neighbor handling is performed internally by the extraction tool.
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            data = data_extraction_tool.data_extraction(ctrl_file_path, "var")
            return np.asarray(data)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            is_transient_dap_error = (
                "DAP failure" in message
                or "NetCDF" in message
                or "HTTP" in message
                or "timeout" in message.lower()
                or "tempor" in message.lower()
                or "connection reset" in message.lower()
            )
            if attempt >= max_retries or not is_transient_dap_error:
                break
            time.sleep(retry_delay_seconds * attempt)

    raise RuntimeError(
        f"ParFlow discharge retrieval failed after {max_retries} attempts for {ctrl_file_path}."
    ) from last_error


def get_static_alluvium_indicator_context(
    alluvium_mask_file,
    indicator_file,
):
    """Load static alluvium and indicator context used for candidate pre-selection."""
    with xr.open_dataset(alluvium_mask_file, decode_times=False) as ds_alluv:
        alluv_lon = ds_alluv["lon"].values
        alluv_lat = ds_alluv["lat"].values
        mask_da = ds_alluv["mask_var"]
        if "time" in mask_da.dims:
            mask_da = mask_da.isel(time=0)
        if "z" in mask_da.dims:
            mask_da = mask_da.isel(z=0)
        mask2d = mask_da.values

    alluv_valid_geo = np.isfinite(alluv_lon) & np.isfinite(alluv_lat)
    alluv_valid_mask = np.isfinite(mask2d)
    fill_value = mask_da.attrs.get("_FillValue", None)
    missing_value = mask_da.attrs.get("missing_value", None)
    if fill_value is not None:
        alluv_valid_mask &= mask2d != fill_value
    if missing_value is not None:
        alluv_valid_mask &= mask2d != missing_value
    alluv_valid = alluv_valid_geo & alluv_valid_mask

    mask_vals = mask2d[alluv_valid]
    unique_vals, counts = np.unique(mask_vals, return_counts=True)
    if len(unique_vals) == 2 and set(np.round(unique_vals).astype(int)) <= {0, 1}:
        river_class = unique_vals[np.argmin(counts)]
        river_cells = (mask2d == river_class) & alluv_valid
    else:
        river_cells = (mask2d > 0) & alluv_valid

    alluv_lons_rad = np.deg2rad(alluv_lon)
    alluv_lats_rad = np.deg2rad(alluv_lat)

    with xr.open_dataset(indicator_file, decode_times=False) as ds_indicator:
        ind_lon = ds_indicator["lon"].values
        ind_lat = ds_indicator["lat"].values

    ind_lons_rad = np.deg2rad(ind_lon)
    ind_lats_rad = np.deg2rad(ind_lat)

    return {
        "alluv_lon": alluv_lon,
        "alluv_lat": alluv_lat,
        "alluv_valid": alluv_valid,
        "river_cells": river_cells,
        "alluv_lons_rad": alluv_lons_rad,
        "alluv_lats_rad": alluv_lats_rad,
        "ind_lon": ind_lon,
        "ind_lat": ind_lat,
        "ind_lons_rad": ind_lons_rad,
        "ind_lats_rad": ind_lats_rad,
    }


def build_alluvium_candidate_context(plant_lats, plant_lons, alluvium_mask_file, indicator_file):
    """Build candidate cell indices for each plant with alluvium preference."""
    data_extraction_tool = _import_data_extraction_tool()

    static_context = get_static_alluvium_indicator_context(
        alluvium_mask_file=alluvium_mask_file,
        indicator_file=indicator_file,
    )

    alluv_valid = static_context["alluv_valid"]
    river_cells = static_context["river_cells"]
    alluv_lons_rad = static_context["alluv_lons_rad"]
    alluv_lats_rad = static_context["alluv_lats_rad"]
    ind_lon = static_context["ind_lon"]
    ind_lat = static_context["ind_lat"]
    ind_lons_rad = static_context["ind_lons_rad"]
    ind_lats_rad = static_context["ind_lats_rad"]

    def _target_and_neighbour_indices(target_idx, grid_shape):
        target_y, target_x = target_idx
        candidate_indices = [(target_y, target_x)]
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                if delta_y == 0 and delta_x == 0:
                    continue
                cand_y = target_y + delta_y
                cand_x = target_x + delta_x
                if 0 <= cand_y < grid_shape[0] and 0 <= cand_x < grid_shape[1]:
                    candidate_indices.append((cand_y, cand_x))
        return candidate_indices

    per_plant_candidate_indices = []
    per_plant_has_river_candidates = []
    nearest_local_idx_per_plant = []

    for i in range(len(plant_lats)):
        plant_lon_rad = np.deg2rad(float(plant_lons[i]))
        plant_lat_rad = np.deg2rad(float(plant_lats[i]))

        dist_ind = data_extraction_tool.spher_dist(ind_lons_rad, ind_lats_rad, plant_lon_rad, plant_lat_rad)
        mapped_idx = np.unravel_index(np.argmin(dist_ind, axis=None), dist_ind.shape)
        candidate_indices = _target_and_neighbour_indices(mapped_idx, dist_ind.shape)

        river_candidates = []
        for cand_y, cand_x in candidate_indices:
            cand_lon_rad = np.deg2rad(ind_lon[cand_y, cand_x])
            cand_lat_rad = np.deg2rad(ind_lat[cand_y, cand_x])
            dist_alluv = data_extraction_tool.spher_dist(alluv_lons_rad, alluv_lats_rad, cand_lon_rad, cand_lat_rad)
            dist_alluv_valid = np.where(alluv_valid, dist_alluv, np.inf)
            alluv_y, alluv_x = np.unravel_index(np.argmin(dist_alluv_valid, axis=None), dist_alluv_valid.shape)
            if river_cells[alluv_y, alluv_x]:
                river_candidates.append((cand_y, cand_x))

        if len(river_candidates) > 0:
            use_candidates = river_candidates
            per_plant_has_river_candidates.append(True)
        else:
            use_candidates = candidate_indices
            per_plant_has_river_candidates.append(False)

        per_plant_candidate_indices.append(use_candidates)

        cand_lons = np.array([float(ind_lon[cy, cx]) for cy, cx in use_candidates])
        cand_lats = np.array([float(ind_lat[cy, cx]) for cy, cx in use_candidates])
        cand_dist = data_extraction_tool.spher_dist(np.deg2rad(cand_lons), np.deg2rad(cand_lats), plant_lon_rad, plant_lat_rad)
        nearest_local_idx_per_plant.append(int(np.argmin(cand_dist)))

    return {
        "ind_lon": ind_lon,
        "ind_lat": ind_lat,
        "per_plant_candidate_indices": per_plant_candidate_indices,
        "selected_from_alluvium": np.array(per_plant_has_river_candidates, dtype=bool),
        "nearest_local_idx_per_plant": np.array(nearest_local_idx_per_plant, dtype=int),
    }


def extract_selected_discharge_alluvium(
    year,
    placements,
    root_dir,
    alluvium_mask_file,
    indicator_file,
    fallback_mode="max_annual", 
):
    """Extract discharge from alluvium-aware candidate cells and select one series per plant."""
    '''
    - only consider all the 9 grid cells (nearest + 8 neighbours) closest to the plant location
    - first check which of the 9 grid cells are on the alluvium river mask
    - if river cells exist: only extract those river cells, then keep discharge from the nearest river cell
    - if none are river cells: extract all 9 cells, 
        and then one can specify in the fallback mode to keep max annual discharge (max_annual) 
        or nearest cell (nearest), default is max_annual
    '''
    # assure hydropower plant identifiers
    if "hydro_plant_id" not in placements.columns:
        placements = placements.copy()
        placements["hydro_plant_id"] = "plant_" + placements.index.astype(str)

    # extract location lat/lon from placements dataframe
    plant_lats = placements["lat"].values
    plant_lons = placements["lon"].values
    
    # --- Build candidate context ---
    # For each plant, compute a short list of nearby indicator-grid candidate
    # cells and mark whether any candidate maps to an alluvium/river cell.
    candidate_context = build_alluvium_candidate_context(
        plant_lats=plant_lats,
        plant_lons=plant_lons,
        alluvium_mask_file=alluvium_mask_file,
        indicator_file=indicator_file,
    )

    ind_lon = candidate_context["ind_lon"]
    ind_lat = candidate_context["ind_lat"]
    per_plant_candidate_indices = candidate_context["per_plant_candidate_indices"]
    selected_from_alluvium = candidate_context["selected_from_alluvium"]
    nearest_local_idx_per_plant = candidate_context["nearest_local_idx_per_plant"]

    extraction_lats = []
    extraction_lons = []
    per_plant_counts = []
    for candidate_indices in per_plant_candidate_indices:
        per_plant_counts.append(len(candidate_indices))
        for cand_y, cand_x in candidate_indices:
            extraction_lats.append(float(ind_lat[cand_y, cand_x]))
            extraction_lons.append(float(ind_lon[cand_y, cand_x]))

    extraction_lats = np.array(extraction_lats)
    extraction_lons = np.array(extraction_lons)
    # --- Retrieve discharge series for all candidates ---
    # We flatten all candidate locations across plants and request ParFlow
    # discharge series for each. The returned array has shape
    # (total_candidates, n_time).
    discharge_info = retrieve_discharge_data(
        year=year,
        latitudes=extraction_lats,
        longitudes=extraction_lons,
        root_dir=root_dir,
        include_neighbours=False,
    )
    # Mask ParFlow missing-value sentinel (large fill value ~= 1e20).
    discharge_info = np.ma.masked_where(np.isclose(discharge_info, 1.0e20), discharge_info)

    n_plants = len(plant_lats)
    n_time = discharge_info.shape[1]
    selected_discharge = np.ma.empty((n_plants, n_time), dtype=discharge_info.dtype)
    selected_candidate_idx = np.zeros(n_plants, dtype=int)

    row_start = 0
    selected_cell_overview = []
    for plant_idx, n_candidates in enumerate(per_plant_counts):
        row_end = row_start + n_candidates
        plant_candidate_data = discharge_info[row_start:row_end, :]
        # --- Select best candidate for this plant ---
        # Priority rules:
        # 1) If any candidate maps to an alluvium/river cell, pick the nearest
        #    among those (useful when river proximity matters). 
        #    However, if the nearest discharge is zero or masked, choose the one with the largest total annual discharge instead.
        # 2) Else if fallback_mode == "nearest", pick the geographically
        #    nearest candidate.
        # 3) Otherwise (default "max_annual"), pick the candidate with the
        #    largest total (annual) discharge.
        if selected_from_alluvium[plant_idx]:
            local_best_idx = nearest_local_idx_per_plant[plant_idx]
            if (plant_candidate_data[local_best_idx, :].sum() == 0) or (plant_candidate_data[local_best_idx, :].sum() is np.ma.masked):
                # If the nearest alluvium candidate has zero discharge or masked values, choose the one with the largest total annual discharge instead.
                annual_discharge = np.ma.filled(np.ma.sum(plant_candidate_data, axis=1), fill_value=-np.inf)
                local_best_idx = int(np.argmax(annual_discharge))
        elif fallback_mode == "nearest":
            local_best_idx = nearest_local_idx_per_plant[plant_idx]
        else:
            # Sum over time for each candidate, filling masked rows with -inf
            # so they are not chosen if all values are masked.
            annual_discharge = np.ma.filled(np.ma.sum(plant_candidate_data, axis=1), fill_value=-np.inf)
            local_best_idx = int(np.argmax(annual_discharge))

        # Store selection and advance to next plant's candidate block.
        selected_candidate_idx[plant_idx] = local_best_idx
        selected_discharge[plant_idx, :] = plant_candidate_data[local_best_idx, :]

        # store metadata for selected candidate
        best_y, best_x = per_plant_candidate_indices[plant_idx][local_best_idx]
        selected_cell_overview.append(
            {
                "plant_id": str(placements.iloc[plant_idx]["hydro_plant_id"]),
                "plant_lon": float(placements.iloc[plant_idx]["lon"]),
                "plant_lat": float(placements.iloc[plant_idx]["lat"]),
                "selected_grid_y": int(best_y),
                "selected_grid_x": int(best_x),
                "selected_grid_lon": float(ind_lon[best_y, best_x]),
                "selected_grid_lat": float(ind_lat[best_y, best_x]),
                "selected_local_candidate_idx": int(local_best_idx),
                "n_candidates_considered": int(n_candidates),
                "selected_from_alluvium_prefilter": bool(selected_from_alluvium[plant_idx]),
            }
        )

        row_start = row_end

    return {
        "selected_discharge_m3_per_day": selected_discharge,
        "selected_candidate_idx": selected_candidate_idx,
        "selected_from_alluvium": selected_from_alluvium,
        "selected_cell_overview": selected_cell_overview,
    }