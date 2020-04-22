import geokit as gk
import reskit as rk
from reskit import windpower

import pandas as pd
import numpy as np
from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple


# ZOOM = 8
# ESACCI = environ["ESACCI_2018_V2_1_1"]

# # TODO: Put GWA into initializer
# GWA = "/data/s-ryberg/backup/weather/global_wind_atlas/WS_100m_global_wgs84_mean_trimmed_europe.tif"


# def ERA5_DIR(x, y, yr):
#     return join(environ["ERA5_PROCESSED_DIR"], "4", str(x), str(y), str(yr))


# def tile_to_tile(xi, yi, zoom, target_zoom):
#     from smopy import num2deg, deg2num

#     lat, lon = num2deg(xi, yi, zoom)
#     xi_out, yi_out = deg2num(lat, lon, target_zoom)
#     return xi_out, yi_out


# def load_weather_data(locs, year, xi, yi):
#     xi, yi = tile_to_tile(xi, yi, ZOOM, target_zoom=4)

#     ext = gk.Extent.fromLocationSet(locs)

#     src_era5 = rk.weather.sources.Era5Source(
#         ERA5_DIR(xi, yi, year), bounds=ext)
#     src_era5.loadSet_Wind()
#     src_era5.loadTemperature(air=True)
#     src_era5.loadPressure()

#     sim_data = OrderedDict()
#     sim_data['times'] = src_era5.timeindex

#     for var in ['air_temp', 'pressure', 'windspeed', ]:
#         tmp = src_era5.get(var, locs, interpolation="cubic",
#                            forceDataFrame=True)
#         sim_data[var] = tmp.reindex(sim_data['times'], method='ffill')

#     # Correct wind speed
#     sim_data['windspeed'][sim_data['windspeed'] < 0] = 0

#     # LRA correction
#     gwa = gk.raster.interpolateValues(GWA, locs)
#     era5_lra = gk.raster.interpolateValues(rk.weather.sources.Era5Source.LONG_RUN_AVERAGE_WINDSPEED_100M,
#                                            locs,
#                                            interpolation="cubic-spline")
#     sim_data['windspeed'] *= gwa/era5_lra

#     return sim_data


# def get_roughness(locs, ext):
#     locs = gk.LocationSet(locs)
#     esa_cci_ras = ext.rasterMosaic(ESACCI)
#     lc_vals = gk.raster.interpolateValues(esa_cci_ras, locs)

#     return rk.windpower.windutil.roughnessFromLandCover(lc_vals, lctype="cci")


# def onshore_wind_from_era5(year, xi, yi, df):

#     # Make sure simulation points is okay (should be a DataFrame)
#     assert 'lon' in df.columns
#     assert 'lat' in df.columns
#     assert 'capacity' in df.columns
#     assert 'hubHeight' in df.columns
#     assert 'rotordiam' in df.columns

#     ext = gk.Extent.fromTile(xi, yi, ZOOM).pad(5, percent=True)

#     locs = gk.LocationSet(df[['lon', 'lat']].values)

#     # Load weather data
#     if not 'roughness' in df.columns:
#         df['roughness'] = get_roughness(locs, ext)

#     weather_data = load_weather_data(locs,
#                                      year,
#                                      xi,
#                                      yi)

#     # TODO: Add pressure correction!

#     # Convolute power curves
#     df['sp_power'] = df['capacity']*1000 / \
#         (np.pi*np.power(df['rotordiam']/2, 2))
#     df['sp_power_sim'] = np.round(df['sp_power']/5)*5

#     # Do simulations of specific power groups
#     gen = pd.DataFrame(np.nan,
#                        index=weather_data['times'],
#                        columns=locs)

#     for sim_value in df['sp_power_sim'].unique():
#         sel = df['sp_power_sim'].values == sim_value
#         df_ = df[sel]

#         # Make Power curve
#         pc = rk.windpower.SyntheticPowerCurve(specificCapacity=sim_value)
#         pc = rk.windpower.convolutePowerCurveByGuassian(pc,
#                                                         stdScaling=0.06,
#                                                         stdBase=0.1,)

#         # Do simulation
#         gen_ = rk.windpower.simulateTurbine(
#             weather_data['windspeed'].values[:, sel],
#             powerCurve=pc,
#             capacity=df_['capacity'].values,
#             rotordiam=df_['rotordiam'].values,
#             hubHeight=df_['hubHeight'].values,
#             measuredHeight=100,
#             roughness=df_['roughness'].values,
#             loss=0.00)
#         gen.values[:, sel] = gen_.values

#     # power sim correction
#     gen = rk.windpower.lowGenCorrection(gen, base=0, sharpness=5)

#     # Done!
#     return gen

Variable = namedtuple("Variable", "name key height")

class WorkflorGenerator():
    def __init__(self, placements):
        # arrange placements, locs, and extent
        assert isinstance(placements, pd.DataFrame)
        self.placements = placements
        self.locs = None

        if 'geom' in placements.columns:
            self.locs = gk.LocationSet( placements.geom )
            self.placements['lon'] = self.locs.lons
            self.placements['lat'] = self.locs.lats
            del self.placements['geom']

        assert 'lon' in placements.columns
        assert 'lat' in placements.columns

        if self.locs is None:
            self.locs = gk.LocationSet( self.placements )

        self.ext = gk.Extent.fromLocationSet(self.locs)

        # Initialize simulation data
        self.sim_data = OrderedDict()
        self.variables = None
        self.sources = OrderedDict()
        self.main_source = None
        self.source_interpolation_mode = OrderedDict()

    ## STAGE 1: configuring
    def for_onshore_wind_energy(self, synthetic_power_curve_cut_out=25, synthetic_power_curve_rounding=1):
        from reskit import windpower

        ## Check for basics
        assert 'capacity' in self.placements.columns, "Placement dataframe needs 'capacity' column"
        assert 'hubHeight' in self.placements.columns, "Placement dataframe needs 'hubHeight' column"

        ## Check for power curve. If not found, make it!
        self.powerCurveLibrary = dict()

        # Should we automatically generate synthetic power curves?
        if not "powerCurve" in self.placements.columns:
            assert 'rotordiam' in self.placements.columns, "Placement dataframe needs 'rotordiam' or 'powerCurve' column"
            
            specificPower = windpower.specificPower(self.placements['capacity'],
                                                    self.placements['rotordiam'])

            if synthetic_power_curve_rounding is not None:
                specificPower = np.round(specificPower/synthetic_power_curve_rounding)*synthetic_power_curve_rounding
                specificPower = specificPower.astype(int)
            
            powerCurve = []
            for sppow in specificPower:
                pcid = "SPC:%d,%d" %(sppow, synthetic_power_curve_cut_out)
                powerCurve.append( pcid )
            
            self.placements['powerCurve'] = powerCurve
        
        # Put power curves into the power curve library
        for pc in self.placements.powerCurve.values:
            assert isinstance(pc, str), "Power curve value needs to be a string, not " + type(pc) 
            
            if pc in self.powerCurveLibrary: continue
            
            if pc[:4] = "SPC:":
                pc = pc.split(":")[1]
                sppow, cutoff = pc.split(",")
                self.powerCurveLibrary[pc] = windpower.SyntheticPowerCurve( 
                                                specificCapacity=float(sppow), 
                                                cutout=float(cutout) )
            else:
                self.powerCurveLibrary[pc] = windpower.TurbineLibrary[pc].PowerCurve
        
        return self
        
    ## STAGE 2: weather data reading and adjusting
    def with_source(self, source_type, path, interpolation_mode="cubic"):
        if source_type == "ERA5":
            self.sources["ERA5"] = rk.weather.sources.Era5Source(
                path, bounds=self.ext)
        elif source_type == "SARAH":
            self.sources["SARAH"] = rk.weather.sources.SarahSource(
                path, bounds=self.ext)
        elif source_type == "MERRA":
            self.sources["MERRA"] = rk.weather.sources.MerraSource(
                path, bounds=self.ext)
        else:
            raise RuntimeError("Unknown source_type")

        if not "times" in self.sim_data:
            self.sim_data['times'] = src.timeindex
            self.main_source = source_type

        self.source_interpolation_mode[source_type] = interpolation_mode

        return self
    
    def read(self, var, source=None):
        if from_source is None:
            from_source = self.main_source

        self.sim_data[var] = self.sources[source].get(
            var, 
            self.locs,
            interpolation=self.source_interpolation_mode[source],
            forceDataFrame=True)

        if not source == self.main_source:
            self.sim_data[var] = self.sim_data[var].reindex(
                self.sim_data['times'], 
                method='ffill')
        
        return self

    ## Stage 3: Weather data adjusting & other intermediate steps 
    def apply_long_run_average(self, variable, source_lra, real_lra, interpolation="cubic-spline"):

        real_lra = gk.raster.interpolateValues(real_lra, self.locs)
        assert np.isnan(real_lra).any() and (real_lra>0).all()

        source_lra = gk.raster.interpolateValues(source_lra, self.locs)
        assert np.isnan(source_lra).any() and (source_lra > 0).all()
        
        self.sim_data[var] *= real_lra/source_lra
        return self

    def estimate_roughness_from_land_cover(self, path, source_type):
        num = gk.raster.interpolateValues(path, self.locs, mode='near')
        self.roughness = rk.util.windutil.roughnessFromLandCover(num, source_type)
        return self

    def project_wind_speeds_to_hub_height_with_log_law(self):
        assert "roughness" in self.placements.columns
        self.sim_data['windspeed'] = rk.weather.windutil.projectByLogLaw(
                                        self.sim_data['wind_speed'],
                                        measuredHeight=self.wind_speed_height, 
                                        targetHeight=self.placements['hubHeight'].values,
                                        roughness=self.placements['roughness'].values
                                        )
        return self

