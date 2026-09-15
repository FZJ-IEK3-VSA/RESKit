"""RESKit example: Northern Germany and the German North Sea.

This example shows the full chain for four technologies:

1. Build the simulation area (the northern German states and the North Sea).
2. Clip the placement databases to this area.
3. Simulate each technology with the related RESKit workflow.
4. Calculate the LCOE of each placement.
5. Plot the LCOE of each placement on a map.

Technologies:
    * Onshore wind   - reskit.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025
    * Offshore wind  - reskit.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025
    * Openfield PV   - reskit.solar.openfield_pv_era5
    * Geothermal EGS - reskit.geothermal.egs_workflow

Run the file as a script, or run the cells one by one in an editor.
"""

# %%
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from shapely.ops import unary_union

import reskit as rk
from reskit.util.local_values import distance_to_coastline, water_depth_from_location

########################################################################
#################### CONFIGURATION #####################################
########################################################################

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(os.path.dirname(HERE), "input_data")
OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEATHER_YEAR = 2018

# The maximum number of placements for each technology. The example takes a
# regular sample of the clipped placements. Set to None to simulate all of them.
MAX_PLACEMENTS = 300

# The northern German states. These states build the onshore simulation area.
NORTHERN_GERMAN_STATES = [
    "Schleswig-Holstein",
    "Hamburg",
    "Bremen",
    "Niedersachsen",
    "Mecklenburg-Vorpommern",
]

# The offshore simulation area is the North Sea part of the German Exclusive
# Economic Zone (EEZ). The German EEZ has two parts. The example keeps the part
# west of this longitude, and drops the Baltic Sea part.
NORTH_SEA_MAX_LON = 10.0

# Geographic data.
PATH_EEZ = (
    "/projects4/2021-p-dunkel-phd/44_shared/forMuhammad/europe_EEZ/"
    "reproject_european_eez_clipped_new.shp"
)  # Marine Regions World EEZ, clipped to Europe
PATH_GADM_LEVEL1 = (
    "/fast/central/shared_data/2026_modelBuilder/regions/"
    "gadm36_GID1(0)_EastWest_and_largeRegion_split_epsg4326_withUnion_v2_1.shp"
)
PATH_WATER_DEPTH = (
    "/fast/central/shared_data/GEBCO_GeneralBathymetricChartOfTheOceans_GEBCO_2025/"
    "gebco_2025/combined/gebco_2025_n90.0_s-90.0_w-180.0_e180.0.tif"
)
PATH_COAST_DISTANCE = (
    "/fast/central/shared_data/NASA_DistanceToNearestCoast_v200906/"
    "GMT_intermediate_coast_distance_01d.tif"
)

# Weather and long-run-average data.
PATH_ERA5 = (
    "/fast/central/shared_data/weather_data/processed_weather_data/"
    "ERA5_global_processed_V2022.02/4/<X-TILE>/<Y-TILE>/"
    f"{WEATHER_YEAR}/reanalysis-era5-single-levels.z4.x<X-TILE>.y<Y-TILE>"
    f".y{WEATHER_YEAR}.*.nc"
)
PATH_GWA_100M = (
    "/benchtop/internal/home/c-winkler/Research/01_Dissertation/03_RESkit/"
    "01_preprocessing/01_expand_GWA_to_EEZ/"
    "01_avg_annual_windspeed_100m_GWA_ERA5_interpolated.tif"
)
GWA_HEIGHT_SCALING_DATA = {
    10: "/fast/central/shared_data/Global_Wind_Atlas/GWA_4.0/wind_speed_cog_10m.tif",
    50: "/fast/central/shared_data/Global_Wind_Atlas/GWA_4.0/wind_speed_cog_50m.tif",
    100: "/fast/central/shared_data/Global_Wind_Atlas/GWA_4.0/wind_speed_cog_100m.tif",
    150: "/fast/central/shared_data/Global_Wind_Atlas/GWA_4.0/wind_speed_cog_150m.tif",
    200: "/fast/central/shared_data/Global_Wind_Atlas/GWA_4.0/wind_speed_cog_200m.tif",
}
PATH_GSA_GHI = (
    "/fast/central/shared_data/2023_gears/geography/irradiance/"
    "global_solar_atlas_v2.9/World_GHI_GISdata_LTAy_AvgDailyTotals_"
    "GlobalSolarAtlas-v2_GEOTIFF/GHI.tif"
)
PATH_GSA_DNI = (
    "/fast/central/shared_data/2023_gears/geography/irradiance/"
    "global_solar_atlas_v2.9/World_DNI_GISdata_LTAy_AvgDailyTotals_"
    "GlobalSolarAtlas-v2_GEOTIFF/DNI.tif"
)

# Economic assumptions. RESKit gives a cost model for wind only. The example
# uses a specific CAPEX for PV. The geothermal workflow gives the LCOE directly.
DISCOUNT_RATE = 0.08
ECONOMIC_ASSUMPTIONS = {
    "onshore_wind": {"lifetime": 25, "opex_per_capex": 0.02},
    "offshore_wind": {"lifetime": 25, "opex_per_capex": 0.03},
    "openfield_pv": {"lifetime": 25, "opex_per_capex": 0.02},
}
OFFSHORE_BASE_SPECIFIC_CAPEX = 2300.0  # [EUR/kW] reference offshore CAPEX
PV_SPECIFIC_CAPEX = 400.0  # [EUR/kW] openfield PV system cost

# The map layout. Four technologies fit into a 2 x 2 grid of maps.
MAP_ROWS, MAP_COLS = 2, 2


# %%
########################################################################
#################### 1. SIMULATION AREA ################################
########################################################################


def build_simulation_area():
    """Build the simulation area as a GeoDataFrame with two parts.

    The first part is the land of the northern German states. The second part
    is the North Sea part of the German EEZ.

    Returns
    -------
    geopandas.GeoDataFrame
        The area with the columns 'name' and 'geometry' in EPSG:4326.
    """
    # Read only the features in the bounding box of the area. The GADM file is
    # large, therefore the bbox filter keeps the read fast.
    gadm = gpd.read_file(PATH_GADM_LEVEL1, bbox=(3.0, 52.0, 15.0, 56.5))
    states = gadm[(gadm["GID_0"] == "DEU") & (gadm["NAME_1"].isin(NORTHERN_GERMAN_STATES))]
    assert len(states) == len(NORTHERN_GERMAN_STATES), "Not all northern German states were found."
    land = states.geometry.union_all()

    # Read the German EEZ. The EEZ holds the North Sea and the Baltic Sea in one
    # polygon, because both seas touch the German coast.
    eez = gpd.read_file(PATH_EEZ).to_crs("EPSG:4326")
    german_eez = eez[eez["ISO_SOV1"] == "DEU"].geometry.union_all()

    # Remove all land from the EEZ. The EEZ reaches the coastline, therefore
    # this step keeps the sea part and the land part of the area apart. The
    # Jutland peninsula also splits the EEZ into the North Sea and the Baltic
    # Sea then.
    water = german_eez.difference(gadm.geometry.union_all())

    # Keep the North Sea parts only.
    parts = [part for part in water.geoms if part.centroid.x < NORTH_SEA_MAX_LON]
    assert parts, "No North Sea part of the German EEZ was found."
    sea = unary_union(parts)

    area = gpd.GeoDataFrame(
        {"name": ["Northern Germany", "German North Sea EEZ"]},
        geometry=[land, sea],
        crs="EPSG:4326",
    )
    return area


area = build_simulation_area()
print(area)


# %%
########################################################################
#################### 2. LOAD AND CLIP THE PLACEMENTS ###################
########################################################################


def to_geodataframe(df):
    """Convert a placement table with 'lon' and 'lat' columns to a GeoDataFrame."""
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )


def clip_placements(df, area, max_placements=MAX_PLACEMENTS):
    """Clip placements to the simulation area and take a regular sample.

    Parameters
    ----------
    df : pandas.DataFrame
        The placements. Needs the columns 'lon' and 'lat'.
    area : geopandas.GeoDataFrame
        The simulation area.
    max_placements : int or None
        The maximum number of placements to keep.

    Returns
    -------
    pandas.DataFrame
        The clipped placements with a reset index.
    """
    gdf = to_geodataframe(df)
    clipped = gpd.clip(gdf, area)
    print(f"Clipped {len(df)} placements to {len(clipped)} placements in the area.")

    if max_placements is not None and len(clipped) > max_placements:
        # Take a regular sample. This keeps the spatial spread of the placements.
        step = len(clipped) // max_placements
        clipped = clipped.iloc[::step].iloc[:max_placements]
        print(f"Sampled {len(clipped)} placements.")

    out = pd.DataFrame(clipped.drop(columns="geometry"))
    out = out.loc[:, ~out.columns.str.startswith("Unnamed")]
    return out.reset_index(drop=True)


def read_geothermal_placements(path, bounds, chunksize=500_000):
    """Read the global geothermal placements in chunks and keep the bounding box.

    Parameters
    ----------
    path : str
        The path to the geothermal placement CSV file.
    bounds : tuple
        The bounding box as (min_lon, min_lat, max_lon, max_lat).
    chunksize : int
        The number of rows for each chunk.

    Returns
    -------
    pandas.DataFrame
        The placements inside the bounding box.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        mask = chunk["lon"].between(min_lon, max_lon) & chunk["lat"].between(min_lat, max_lat)
        chunks.append(chunk[mask])
    return pd.concat(chunks, ignore_index=True)


placements = {}

placements["onshore_wind"] = clip_placements(
    pd.read_csv(os.path.join(INPUT_DIR, "trep-db/WindOnshore/S2_Expansive/Municipalities/Capacities.csv")),
    area,
)
placements["offshore_wind"] = clip_placements(
    pd.read_csv(os.path.join(INPUT_DIR, "trep-db/WindOffshore/S1_Expansive/EEZ/Capacities.csv")),
    area,
)
placements["openfield_pv"] = clip_placements(
    pd.read_csv(os.path.join(INPUT_DIR, "trep-db/OpenfieldPV/S3_Combination/Municipalities/Capacities.csv")),
    area,
)
placements["geothermal"] = clip_placements(
    read_geothermal_placements(
        os.path.join(INPUT_DIR, "geothermal/04_allPlacements.csv"),
        bounds=area.total_bounds,
    ),
    area,
)

for technology, df in placements.items():
    print(f"{technology:15s}: {len(df):5d} placements")


# %%
########################################################################
#################### 3. SIMULATE #######################################
########################################################################

results = {}

# --- Onshore wind ---------------------------------------------------------
results["onshore_wind"] = rk.execute_workflow_iteratively(
    workflow=rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025,
    weather_path_varname="era5_path",
    zoom=4,
    placements=placements["onshore_wind"],
    era5_path=PATH_ERA5,
    gwa_100m_path=PATH_GWA_100M,
    height_scaling_data=GWA_HEIGHT_SCALING_DATA,
    max_batch_size=20000,
)

# %%
# --- Offshore wind --------------------------------------------------------
# The offshore turbines use the same workflow. The workflow covers onshore and
# offshore locations.
results["offshore_wind"] = rk.execute_workflow_iteratively(
    workflow=rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025,
    weather_path_varname="era5_path",
    zoom=4,
    placements=placements["offshore_wind"],
    era5_path=PATH_ERA5,
    gwa_100m_path=PATH_GWA_100M,
    height_scaling_data=GWA_HEIGHT_SCALING_DATA,
    max_batch_size=20000,
)

# %%
# --- Openfield PV ---------------------------------------------------------
results["openfield_pv"] = rk.execute_workflow_iteratively(
    workflow=rk.solar.openfield_pv_era5,
    weather_path_varname="era5_path",
    zoom=4,
    placements=placements["openfield_pv"],
    era5_path=PATH_ERA5,
    global_solar_atlas_ghi_path=PATH_GSA_GHI,
    global_solar_atlas_dni_path=PATH_GSA_DNI,
)

# %%
# --- Geothermal EGS -------------------------------------------------------
# The geothermal workflow reads its own data. It does not need weather data.
results["geothermal"] = rk.geothermal.egs_workflow(placements=placements["geothermal"])


# %%
########################################################################
#################### 4. LCOE ###########################################
########################################################################


def mean_capacity_factor(dataset):
    """Return the mean capacity factor of each placement as a numpy array."""
    return dataset["capacity_factor"].mean(dim="time").values


def annual_production_kWh(capacity_kW, mean_cf):
    """Return the mean annual production of each placement in kWh."""
    return capacity_kW * mean_cf * 8760.0


def lcoe_onshore_wind(dataset):
    """Calculate the LCOE of the onshore turbines in EUR/kWh.

    The CAPEX comes from the RESKit onshore cost and scaling model. The model
    scales the cost of a baseline turbine with capacity, hub height and rotor
    diameter.
    """
    capacity = dataset["capacity"].values  # [kW]
    capex = rk.wind.onshore_turbine_capex(
        capacity=capacity,
        hub_height=dataset["hub_height"].values,
        rotor_diam=dataset["rotor_diam"].values,
    )  # [EUR]
    production = annual_production_kWh(capacity, mean_capacity_factor(dataset))
    return rk.util.levelized_cost_of_electricity_simplified(
        capex=capex,
        mean_production=production,
        discount_rate=DISCOUNT_RATE,
        **ECONOMIC_ASSUMPTIONS["onshore_wind"],
    )


def lcoe_offshore_wind(dataset):
    """Calculate the LCOE of the offshore turbines in EUR/kWh.

    The RESKit offshore cost model scales a reference specific CAPEX with the
    water depth and the distance to the coast of each placement.
    """
    lon = dataset["lon"].values
    lat = dataset["lat"].values

    water_depth = np.array(
        [
            water_depth_from_location(
                latitude=_lat,
                longitude=_lon,
                waterDepthFilePath=PATH_WATER_DEPTH,
                consider_only="negative",
            )
            for _lon, _lat in zip(lon, lat)
        ],
        dtype=float,
    )
    coast_distance = np.array(
        [
            distance_to_coastline(
                latitude=_lat,
                longitude=_lon,
                distancetoCoastFilePath=PATH_COAST_DISTANCE,
            )
            for _lon, _lat in zip(lon, lat)
        ],
        dtype=float,
    )
    # The NASA raster gives negative values on the sea. Use the magnitude.
    coast_distance = np.abs(coast_distance)

    capacity = dataset["capacity"].values  # [kW]
    specific_capex = rk.wind.calculate_specific_offshore_capex(
        baseSpecCapex=OFFSHORE_BASE_SPECIFIC_CAPEX,
        capacity=capacity,
        rotorDiam=dataset["rotor_diam"].values,
        hubHeight=dataset["hub_height"].values,
        waterDepth=water_depth,
        coastDistance=coast_distance,
        portDistance=coast_distance,
    )  # [EUR/kW]
    production = annual_production_kWh(capacity, mean_capacity_factor(dataset))
    return rk.util.levelized_cost_of_electricity_simplified(
        capex=specific_capex * capacity,
        mean_production=production,
        discount_rate=DISCOUNT_RATE,
        **ECONOMIC_ASSUMPTIONS["offshore_wind"],
    )


def lcoe_openfield_pv(dataset):
    """Calculate the LCOE of the openfield PV systems in EUR/kWh.

    RESKit has no PV cost model. The example uses a constant specific CAPEX.
    """
    capacity = dataset["capacity"].values  # [kW]
    production = annual_production_kWh(capacity, mean_capacity_factor(dataset))
    return rk.util.levelized_cost_of_electricity_simplified(
        capex=PV_SPECIFIC_CAPEX * capacity,
        mean_production=production,
        discount_rate=DISCOUNT_RATE,
        **ECONOMIC_ASSUMPTIONS["openfield_pv"],
    )


def lcoe_geothermal(dataset):
    """Return the LCOE of the EGS placements in EUR/kWh.

    The geothermal workflow calculates the LCOE itself. The example uses the
    Gringarten method at the optimal drilling depth.
    """
    lcoe = dataset["LCOE_GR_EUR_per_kWh"].values.astype(float)
    # The workflow marks placements without a usable resource with infinity.
    lcoe[~np.isfinite(lcoe)] = np.nan
    return lcoe


LCOE_FUNCTIONS = {
    "onshore_wind": lcoe_onshore_wind,
    "offshore_wind": lcoe_offshore_wind,
    "openfield_pv": lcoe_openfield_pv,
    "geothermal": lcoe_geothermal,
}

lcoe = {}
for technology, dataset in results.items():
    values = LCOE_FUNCTIONS[technology](dataset)
    lcoe[technology] = gpd.GeoDataFrame(
        {"lcoe_ct_per_kWh": np.asarray(values, dtype=float) * 100.0},
        geometry=gpd.points_from_xy(dataset["lon"].values, dataset["lat"].values),
        crs="EPSG:4326",
    )
    print(
        f"{technology:15s}: LCOE "
        f"{np.nanmin(lcoe[technology]['lcoe_ct_per_kWh']):6.2f} - "
        f"{np.nanmax(lcoe[technology]['lcoe_ct_per_kWh']):6.2f} ct/kWh"
    )

# Write the results to disk.
for technology, gdf in lcoe.items():
    gdf.to_file(os.path.join(OUTPUT_DIR, f"lcoe_{technology}.gpkg"), driver="GPKG")


# %%
########################################################################
#################### 5. PLOT ###########################################
########################################################################

TITLES = {
    "onshore_wind": "Onshore wind",
    "offshore_wind": "Offshore wind",
    "openfield_pv": "Openfield PV",
    "geothermal": "Geothermal EGS",
}


def plot_lcoe_maps(lcoe, area, path=None):
    """Plot the LCOE of each placement on one map for each technology.

    Parameters
    ----------
    lcoe : dict
        The LCOE GeoDataFrame of each technology.
    area : geopandas.GeoDataFrame
        The simulation area. The function draws it as a background.
    path : str, optional
        The path of the output figure. The function shows the figure if None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    min_lon, min_lat, max_lon, max_lat = area.total_bounds
    # Stretch the latitude axis, so that the maps keep the shape of the area.
    aspect = 1.0 / np.cos(np.deg2rad(0.5 * (min_lat + max_lat)))

    # Size the figure to the shape of the area. The constrained layout then
    # removes the empty space around the maps.
    map_width = 5.5
    map_height = map_width * aspect * (max_lat - min_lat) / (max_lon - min_lon)

    fig, axes = plt.subplots(
        MAP_ROWS,
        MAP_COLS,
        figsize=(MAP_COLS * (map_width + 1.2), MAP_ROWS * (map_height + 1.0)),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for ax, (technology, gdf) in zip(axes.flatten(), lcoe.items()):
        # Draw the simulation area as a background.
        area.plot(ax=ax, facecolor="#eeeeee", edgecolor="#888888", linewidth=0.6, zorder=0)

        values = gdf["lcoe_ct_per_kWh"]
        # Clip the colour range to the 5th and 95th percentile. Single extreme
        # placements do not compress the colour scale then.
        vmin, vmax = np.nanpercentile(values, [5, 95])
        gdf.plot(
            ax=ax,
            column="lcoe_ct_per_kWh",
            cmap="viridis_r",
            markersize=8,
            norm=Normalize(vmin=vmin, vmax=vmax),
            zorder=1,
        )

        ax.set_title(f"{TITLES[technology]}  (n = {len(gdf)})")
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_aspect(aspect)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")

        # Put the colorbar next to the map.
        fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis_r"),
            ax=ax,
            fraction=0.046,
            pad=0.02,
            label="LCOE [ct/kWh]",
        )

    # Hide the axes that hold no technology.
    for ax in axes.flatten()[len(lcoe) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"RESKit LCOE of Northern Germany and the German North Sea ({WEATHER_YEAR})",
        fontsize=14,
    )
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote the figure to: {path}")
    return fig


figure = plot_lcoe_maps(lcoe, area, path=os.path.join(OUTPUT_DIR, "lcoe_maps.png"))
plt.show()

# %%
