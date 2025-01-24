import geokit as gk
import numpy as np
import pandas as pd


def visibility_from_topography(
    lon,
    lat,
    elevation_raster,
    base_elevation=None,
    eye_level=2,
    max_degree=0.1,
    degree_step=0.003,
    theta_step=3,
    _interpolation="cubic",
):
    """Determine visibility around a given point based on topography. Also
    gives planar angle, planar elevation, and planar distance of multiple
    sample points surroudning the specified latitude and longitude


    Params:
    ======
    lon : longitude point to consider
    lat : latitude point to consider
    elevation_raster : The raster to extract elevation values from
    eye_level : The height to measure visibility from (in meters) above the elevation at the center point
    base_elevation : The total height to measure visibility from (in meters)
        - If given, 'eye_level' is ignored
    max_degree : The maximum distance to consider (in degrees)
    degree_step : approximate radial degree discretization
    theta_step : theta discretization

    Return:
    ======
    dict
        - All items are pandas DataFrames
            * Columns match to theta direction (in radians)
        - Items include:
            * latitude : the latitude at each sample point
            * longitude : the longitude at each sample point
            * planar_angle : the angle between the view point and the sample point
            * planar_dist : the planar distance between the view point and the sample point
            * planar_elev : the planar elevation of the sample point
            * visibility : indicates if the sample point should be visible from the view point

    Note:
    =====
    * This algorithm will likely fail if used too near to the poles
    * It is also important to choose an apporpriate degree_step for the
      given elevation raster file. If the step size is too small,
      artifacts will begin to appear in leading to reduced visability
      - Todo: investigate this further?? (It has something to do with
        the interpolation
      - For now, choose a value which is slightly higher than the raster's
        pixel resolution

    * You can plot the outputs nicely with:
       >>> fig = plt.figure()
       >>> ax = fig.add_subplot(111, polar=True)
       >>> h = ax.pcolormesh(a['visibility'].columns,
       >>>                   a['visibility'].index,
       >>>                   a['visibility'])
       >>> plt.colorbar(h)
       >>> plt.show()
    """
    lon_r, lat_r = np.radians([lon, lat])
    theta_step_r = np.radians(theta_step)
    thetas_r = np.arange(0, 2 * np.pi, theta_step_r) + theta_step_r
    thetas = np.degrees(thetas_r)

    approx_center_earth_thetas = (
        np.arange(0, max_degree + degree_step, degree_step) + degree_step / 2
    )
    approx_center_earth_thetas_r = np.radians(approx_center_earth_thetas)

    if base_elevation is None:
        base_elevation = (
            gk.raster.interpolateValues(elevation_raster, (lon, lat)) + eye_level
        )

    # Construct sample points
    # TODO: construct the sample points in a way that ensure equal distances!
    sample_lats_r, sample_lons_r = [], []
    for acetr in approx_center_earth_thetas_r:
        sample_lats_r.append(np.sin(thetas_r) * acetr + lat_r)
        sample_lons_r.append(np.cos(thetas_r) * acetr / np.sin(lat_r) + lon_r)

    sample_lons_r = np.array(sample_lons_r)
    sample_lons = np.degrees(sample_lons_r)
    sample_lats_r = np.array(sample_lats_r)
    sample_lats = np.degrees(sample_lats_r)

    # Compute distance in plane
    R_earth = 6378137.0  # Earth radius at sea level
    Ro = R_earth + base_elevation

    BdotC = np.sin(lat_r) * np.sin(sample_lats_r) * np.cos(
        lon_r - sample_lons_r
    ) + np.cos(lat_r) * np.cos(sample_lats_r)
    center_earth_theta = np.arccos(BdotC)

    locs = np.column_stack(
        [
            sample_lons.flatten(),
            sample_lats.flatten(),
        ]
    )
    elevs = gk.raster.interpolateValues(
        elevation_raster, locs, interpolate=_interpolation
    )
    #     elevs = np.full( locs.shape[0], base_elevation-eye_level)
    elevs = elevs.reshape(sample_lons.shape)

    Rt = elevs + R_earth
    planar_dist = Rt * np.sin(center_earth_theta)
    planar_elev = np.sqrt(np.power(Rt, 2) - np.power(planar_dist, 2)) - Ro
    planar_angle = np.arctan(planar_elev / planar_dist)

    # Determine visibility
    visibility = [np.ones_like(thetas_r)]
    for di in range(1, len(approx_center_earth_thetas_r)):
        visibility.append(planar_angle[di, :] >= (planar_angle[:di, :].max(axis=0)))
    visibility = np.array(visibility)

    # Done!
    col = pd.Series(thetas_r)
    col.name = "theta_direction_rad"

    idx = pd.Series(np.arange(len(approx_center_earth_thetas_r)))
    idx.name = "distance_index"

    return dict(
        latitude=pd.DataFrame(sample_lats, columns=col, index=idx),
        longitude=pd.DataFrame(sample_lons, columns=col, index=idx),
        elevation_angle=pd.DataFrame(planar_angle, columns=col, index=idx),
        planar_dist=pd.DataFrame(planar_dist, columns=col, index=idx),
        #        planar_elevation = pd.DataFrame(planar_elev, columns=col, index=idx),
        #         elevs = pd.DataFrame(elevs, columns=col, index=idx),
        visibility=pd.DataFrame(visibility, columns=col, index=idx),
    )


# EXAMPLE
if __name__ == "__main__":
    ext = gk.Extent.fromTile(66, 45, 7).castTo(4326).pad(0.2)

    # elev_raster = ext.rasterMosaic(ELEV_DATA, pixelWidth=0.0020833333333333333/2, pixelHeight=0.0020833333333333333/2,
    #                               _warpKwargs=dict(resampleAlg='average'))
    elev_raster = ext.rasterMosaic(ELEV_DATA)

    a = visibilityFromTopography(6.952, 46.296, elev_raster, eye_level=20)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    h = ax.pcolormesh(
        a["elevation_angle"].index,
        a["elevation_angle"].columns,
        a["elevation_angle"].values.T,
    )
    plt.colorbar(h)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    h = ax.pcolormesh(
        a["ground_visibility"].index,
        a["ground_visibility"].columns,
        a["ground_visibility"].values.T,
    )
    plt.colorbar(h)
    plt.show()

    print(
        a["ground_visibility"].sum().sum()
        / (a["ground_visibility"].shape[0] * a["ground_visibility"].shape[1])
    )
