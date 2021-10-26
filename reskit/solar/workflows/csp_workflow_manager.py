from logging import error, warn, warning

from ...workflow_manager import WorkflowManager
from .CSP_data.database_loader import load_dataset
from .solar_workflow_manager import SolarWorkflowManager
import numpy as np
import pandas as pd
import pvlib
from numba import jit
import time
import geokit as gk
from typing import Union

class PTRWorkflowManager(SolarWorkflowManager):
    def __init__(self, placements):
        """

        __init_(self, placements)

        Initialization of an instance of the generic SolarWorkflowManager clas

        Parameters
        ----------
        placements : pandas Dataframe
                     The locations that the simulation should be run for.
                     Columns must include "lon", "lat"

        Returns
        -------
        SolarWorkflorManager

        """

        # Do basic workflow construction
        super().__init__(placements)
        self._time_sel_ = None
        self._time_index_ = None
        self.module = None

        self.check_placements()
        assert 'land_area_m2' in placements.columns or \
                'aperture_area_m2' in placements.columns or \
                'area' in placements.columns, 'InputError: Placements need an area column "land_area_m2". Please add a size in m^2 (e.g.: 2.5E6)'
        
    def check_placements(self):
        assert hasattr(self, 'placements')
        assert isinstance(self.placements, pd.DataFrame)
        assert 'lat' in self.placements.columns or 'latitude' in self.placements.columns
        assert 'lon' in self.placements.columns or 'longitude' in self.placements.columns
        assert 'land_area_m2' in self.placements.columns \
            or 'aperture_area_m2' in self.placements.columns \
            or 'area' in self.placements.columns
    
    def loadPTRdata(self, datasetname:str):
        '''loads the dataset with the name datasetname.

        Parameters
        ----------
        datasetname : str
            [description]
        '''
        self.sim_data['ptr_data'] = load_dataset(datasetname=datasetname)
        # make list from coefficients from regression
        self.sim_data['ptr_data']['b'] = np.array([
            self.sim_data['ptr_data']['b0'],
            self.sim_data['ptr_data']['b1'],
            self.sim_data['ptr_data']['b2'],
            self.sim_data['ptr_data']['b3'],
            self.sim_data['ptr_data']['b4'],
        ])
        return self.sim_data['ptr_data']
        
    def determine_area(self):
        '''determines the land area, aperture area from given placement dataframe.
        If only 'area' is given, it will be assumed as land area.
        '''
        assert 'ptr_data' in self.sim_data.keys()
        assert 'SF_density_total' in self.sim_data['ptr_data'].index
        columns = self.placements.columns
        
        #if only area in placements:
        if 'area' in  columns and not 'aperture_area_m2' in columns and not 'land_area_m2' in columns:
            warn('Key "area" is assumed to be the land area. Abort if wrong!')
            self.placements['land_area_m2'] = self.placements['area']
            self.placements.drop('area', axis=1)
            self.placements['aperture_area_m2'] = self.placements['land_area_m2'] * self.sim_data['ptr_data']['SF_density_total']
        
        #only aperture_area_m2 in placements
        elif 'aperture_area_m2' in columns and not 'land_area_m2' in columns:
            self.placements['aperture_area_m2'] = self.placements['land_area_m2'] * self.sim_data['ptr_data']['SF_density_total']
        
        #only land_area_m2 in placements
        elif 'land_area_m2' in columns and not 'aperture_area_m2' in columns:
            self.placements['land_area_m2'] = self.placements['aperture_area_m2'] / self.sim_data['ptr_data']['SF_density_total']                          
    
    def get_timesteps(self):
        self._numtimesteps = self.time_index.shape[0]
        self._numlocations = self.placements.shape[0]
        return self


    def easycalc(self):
        """[do some easy caclution of the heat generation of the solar field based on factors]

        Returns:
            [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for self.sim_data['Heat_kW']]
        """
        area_usage = 0.5
        geometric_efficiency = 0.7

        self.sim_data['Heat_kW'] = np.multiply(self.placements['Area'].to_numpy(), self.sim_data['direct_normal_irradiance']) * area_usage * geometric_efficiency

        return self
    

    # def adjust_variable_to_long_run_average(
    #     self,
    #     variable: str,
    #     source_long_run_average: Union[str, float, np.ndarray],
    #     real_long_run_average: Union[str, float, np.ndarray],
    #     real_lra_scaling: float = 1,
    #     spatial_interpolation: str = "linear-spline"):
        
    #     """Adjusts the average mean of the specified variable to a known long-run-average

    #     Note:
    #     -----
    #     uses the equation: variable[t] = variable[t] * real_long_run_average / source_long_run_average

    #     Parameters
    #     ----------
    #     variable : str
    #         The variable to be adjusted

    #     source_long_run_average : Union[str, float, np.ndarray]
    #         The variable's native long run average (the average in the weather file)
    #         - If a string is given, it is expected to be a path to a raster file which can be 
    #           used to look up the average values from using the coordinates in `.placements`
    #         - If a numpy ndarray (or derivative) is given, the shape must be one of (time, placements)
    #           or at least (placements) 

    #     real_long_run_average : Union[str, float, np.ndarray]
    #         The variables 'true' long run average
    #         - If a string is given, it is expected to be a path to a raster file which can be 
    #           used to look up the average values from using the coordinates in `.placements`
    #         - If a numpy ndarray (or derivative) is given, the shape must be one of (time, placements)
    #           or at least (placements)

    #     real_lra_scaling : float, optional
    #         An optional scaling factor to apply to the values derived from `real_long_run_average`. 
    #         - This is primarily useful when `real_long_run_average` is a path to a raster file
    #         - By default 1

    #     spatial_interpolation : str, optional
    #         When either `source_long_run_average` or `real_long_run_average` are a path to a raster 
    #         file, this input specifies which interpolation algorithm should be used
    #         - Options are: "near", "linear-spline", "cubic-spline", "average"
    #         - By default "linear-spline"
    #         - See for more info: geokit.raster.interpolateValues

    #     Returns
    #     -------
    #     WorkflowManager
    #         Returns the invoking WorkflowManager (for chaining)
    #     """

    #     if isinstance(real_long_run_average, str):
    #         real_lra = gk.raster.interpolateValues(
    #             real_long_run_average,
    #             self.locs,
    #             mode=spatial_interpolation)
    #         assert not np.isnan(real_lra).any() and (real_lra > 0).all()
    #     else:
    #         real_lra = real_long_run_average

    #     if isinstance(source_long_run_average, str):
    #         source_lra = gk.raster.interpolateValues(
    #             source_long_run_average,
    #             self.locs,
    #             mode=spatial_interpolation)
    #         assert not np.isnan(source_lra).any() and (source_lra > 0).all()
    #     else:
    #         source_lra = source_long_run_average

    #     self.sim_data[variable] *= real_lra * real_lra_scaling / source_lra
    #     print('Factors_from_LRA:')
    #     print(real_lra * real_lra_scaling / source_lra)
    #     print('__')
    #     return self

    def direct_normal_irradiance_from_trigonometry(self):
        """

        direct_normal_irradiance_from_trigonometry(self):


        Parameters
        ----------
        None

        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object.

        Notes
        -----
        Required columns in the placements dataframe to use this functions are 'lon', 'lat' and 'elev'.
        Required data in the sim_data dictionary are 'direct_horizontal_irradiance' and 'apparent_solar_zenith'.

        Calculates the direct normal irradiance from the following equation:
            .. math:: dir_nor_irr = dir_hor_irr / cos( solar_zenith )

            Where:
            dir_nor_irr  -> The direct irradiance on the normal plane
            dir_hor_irr  -> The direct irradiance on the horizontal plane
            solar_zenith -> The solar zenith angle in radians

        """

        # TODO: This can also cover the case when we know GHI & DiffHI
        assert "direct_horizontal_irradiance" in self.sim_data
        assert "solar_zenith_degree" in self.sim_data

        dni_flat = self.sim_data['direct_horizontal_irradiance']
        zen = np.radians(self.sim_data['solar_zenith_degree'])

        self.sim_data['direct_normal_irradiance'] = dni_flat / np.cos(zen)
        
        index_out = (dni_flat < 25) & (np.cos(zen) < 0.05)
        self.sim_data['direct_normal_irradiance'][index_out] = 0

        sel = ~np.isfinite(self.sim_data['direct_normal_irradiance'])
        sel = np.logical_or(sel, self.sim_data['direct_normal_irradiance'] < 0)
        sel = np.logical_or(sel, self.sim_data['direct_normal_irradiance'] > 1600)

        self.sim_data['direct_normal_irradiance'][sel] = 0

        return self

    def apply_elevation(self, elev):
        """

        apply_elevation(self)

        Adds an elevation (name: 'elev') column to the placements data frame.

        Parameters
        ----------
        elev: str, list
              If a string is given it must be a path to a rasterfile including the elevations.
              If a list is given it has to include the elevations at each location.


        Returns
        -------
        Returns a reference to the invoking SolarWorkflowManager object

        """

        if 'elev' in self.placements.columns:
            return self

        if elev == None:
            self.placements['elev']=0

        elif isinstance(elev, str):
            clipped_elev = self.ext.pad(0.5).rasterMosaic(elev)
            self.placements['elev'] = gk.raster.interpolateValues(
                clipped_elev,
                self.locs)
        else:
            self.placements['elev'] = elev

        return self


    def calculateSolarPosition(self):
        """calculates the solar position in terms of hour angle and declination from time series and location series of the current object

        Returns:
            [CSPWorkflowManager]: Updated CSPWorkflowManager with new values for sim_data['values'][timeserie_iter, location_iter]. The calculated values are:
                                    - solar_zenith_degree: solar zenith angle
                                    - solar_altitude_angle_degree: solar altitude (elevation) angle in degrees
                                    - aoi_northsouth: angle of incidence for northsouth-orientation of trough
                                    - aoi_eastwest: angle of incidence for eastwest-orientation of trough

        """

        #check for inputs
        assert 'lat' in self.placements.columns
        assert 'lon' in self.placements.columns
        assert 'elev' in self.placements.columns


        #set up empty array
        #self.sim_data['hour_angle_degree'] = np.empty(shape=(self._numtimesteps, self._numlocations))
        #self.sim_data['declination_angle_degree'] = np.empty(shape=(self._numtimesteps, self._numlocations))
        self.sim_data['solar_zenith_degree'] = np.empty(shape=(self._numtimesteps, self._numlocations))
        self.sim_data['aoi_northsouth'] = np.empty(shape=(self._numtimesteps, self._numlocations))
        self.sim_data['aoi_eastwest'] = np.empty(shape=(self._numtimesteps, self._numlocations))




        # iterate trough all location
        for location_iter, row in enumerate(self.placements[['lon', 'lat', 'elev']].itertuples()):
            


            #calculate the solar hour angle with pv lib
            # self.sim_data['hour_angle_degree'][:, location_iter] = \
            #     pvlib.solarposition.hour_angle(
            #         times=self.time_index,
            #         longitude=row.lon,
            #         equation_of_time=pvlib.solarposition.equation_of_time_pvcdrom(dayofyear=self.time_index.day_of_year.values)
            #         )
            #         # equation_of_time is the deviation betwee real time and solar time



            # #calculate the solar declonation angle with pv lib
            # self.sim_data['declination_angle_degree'][:, location_iter] = \
            #     pvlib.solarposition.declination_cooper69(dayofyear=self.time_index.day_of_year.values) * 180 / np.pi


            #calculate the solar position
            _solarpos = \
                pvlib.solarposition.get_solarposition(
                    time=self.time_index,
                    latitude=row.lat,
                    longitude=row.lon,
                    #altitude=row.elev,
                    #pressure=self.sim_data["surface_pressure"][:, location_iter], #TODO: insert here
                    #temperature=self.sim_data["surface_air_temperature"][:, location_iter], #TODO: insert here
                    #method='nrel_numba'
                )


            self.sim_data['solar_zenith_degree'][:, location_iter] = _solarpos['apparent_zenith'].values

            #calculate aoi
            truetracking_angles = pvlib.tracking.singleaxis(
                apparent_zenith=_solarpos['apparent_zenith'],
                apparent_azimuth=_solarpos['azimuth'],
                axis_tilt=0,
                axis_azimuth=180,
                max_angle=90,
                backtrack=False,  # for true-tracking
                gcr=0.5)  # irrelevant for true-tracking

            self.sim_data['aoi_northsouth'][:, location_iter] = np.nan_to_num(truetracking_angles['aoi'].values)

            #calculate aoi
            truetracking_angles = pvlib.tracking.singleaxis(
                apparent_zenith=_solarpos['apparent_zenith'],
                apparent_azimuth=_solarpos['azimuth'],
                axis_tilt=0,
                axis_azimuth=90,
                max_angle=180,
                backtrack=False,  # for true-tracking
                gcr=0.5)  # irrelevant for true-tracking

            self.sim_data['aoi_eastwest'][:, location_iter] = np.nan_to_num(truetracking_angles['aoi'].values)

            #from [1]	KALOGIROU, Soteris A. Environmental Characteristics. In: Soteris Kalogirou, ed. Solar energy engineering. Processes and systems. Waltham, Mass: Academic Press, 2014, pp. 51-123.
            # fromula 2.12
        
        self.sim_data['solar_altitude_angle_degree'] = np.rad2deg(np.arcsin(np.cos(np.deg2rad(self.sim_data['solar_zenith_degree']))))

        return self


    def calculateCosineLossesParabolicTrough(self, orientation: str = 'song2013'):
        """[calculate the cosine losses of a parabolic trough CSP solar field.
        Based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6893622 or
        "SONG, Y. Q., Y. XIANG, Y. B. LIAO, B. ZHANG, L. WU, and H. T. ZHANG. How to decide the alignment of the parabolic
        trough collector according to the local latitude. 2013 International Conference on Materials for Renewable Energy
        and Environment (ICMREE 2013). Chengdu, China, 19 - 21 August 2013 ; [proceedings. Piscataway, NJ: IEEE, 2013, pp. 94-97."]

        Returns:
            [type]: [description]
        """
        
        #spellcheck input
        orientation = orientation.lower()
        assert orientation in ['northsouth', 'eastwest', 'song2013']
        # northsouth: north south
        # eastwest: east west
        # song2013: apply logic form song 2013 to dertermine the orientation of solar fields
        assert 'aoi_northsouth' in self.sim_data.keys()
        assert 'aoi_eastwest' in self.sim_data.keys()
        #assert 'solar_altitude_angle_degree' in self.sim_data.keys()
        assert 'lat' in list(self.placements.columns)




        # write data all locations to one 1-D array
        #_hour_angle = self.sim_data['hour_angle_degree'].flatten(order='F')
        #_declination = self.sim_data['declination_angle_degree'].flatten(order='F')
        #_solar_altitude = self.sim_data['solar_altitude_angle_degree'].flatten(order='F')
        _latitude = np.tile(self.placements['lat'].values, (self.time_index.shape[0],1))#self.placements['lat']#.repeat(self.time_index.shape[0])

        #degree to radians
        #_hour_angle = np.deg2rad(_hour_angle)
        #_declination = np.deg2rad(_declination) 
        #_solar_altitude = np.deg2rad(_solar_altitude)

        #calculate the cos of theta for a north / south oreintation
        #formula (6) from Song How to decide the alignment of the parabolic trough collector according to the local latitude 2013
        
        if orientation in ['northsouth', 'song2013']:
            '''
            Sz = np.cos(_latitude) * np.cos(_declination) * np.cos(_hour_angle) + np.sin(_latitude) * np.sin(_declination)
            Sx = - np.cos(_declination) * np.sin(_hour_angle)
            Sy = np.cos(_latitude) * np.sin(_declination) - np.sin(_latitude) * np.cos(_declination) * np.cos(_hour_angle)

            theta_northsouth = np.arccos(1 - ( np.abs(Sz) / np.sqrt(np.square(Sy) + np.square(Sz))) 
            theta_northsouth = np.rad2deg(theta_northsouth)

            # when sun altidude is negative (after sunset), set values to 0
            costheta_northsouth = costheta_northsouth * (_solar_altitude >0)
            '''

            # costheta_northsouth = 1 - ((np.cos(_latitude) * np.cos(_declination) * np.cos(_hour_angle) + np.sin(_latitude) * np.sin(_declination))\
            #     / np.sqrt( np.square(np.cos(_latitude) * np.sin(_declination) - np.sin(_latitude) * np.cos(_declination) * np.cos(_hour_angle))
            #                 + np.square(np.cos(_latitude) * np.cos(_declination) * np.cos(_hour_angle) +  np.sin(_latitude) * np.sin(_declination))))
            
            theta_northsouth = self.sim_data['aoi_northsouth']#.flatten(order='F')


        #calculate the cos of theta for a east/ west oreintation
        #formula (7) from Song How to decide the alignment of the parabolic trough collector according to the local latitude 2013
        
        if orientation in ['eastwest', 'song2013']:
            
            '''
            #calculation
            theta_eastwest = np.arccos(np.sqrt( (np.square(np.sin(_declination))-1) * np.square(np.sin(_hour_angle)) + 1 ))
            theta_eastwest = np.rad2deg(theta_eastwest)

            # when sun altidude is negative (after sunset), set values to 0
            theta_eastwest = theta_eastwest * (_solar_altitude >0)
            '''

            theta_eastwest = self.sim_data['aoi_eastwest']#.flatten(order='F')


        if orientation == 'northsouth':
            theta = theta_northsouth #forced northsouth orientation
            
        elif orientation == 'eastwest':
            theta = theta_eastwest #forced eastwest orientation
            
        elif orientation == 'song2013':
            # apply song 2013:
            # if latitude is between -46° and +46°, use northsouth orientation, else eastwest orientation
            _isns = np.logical_and(_latitude < 46.06 /180*np.pi, _latitude > -46.06 /180*np.pi)

            theta = _isns * theta_northsouth + np.logical_not(_isns) * theta_eastwest


        # change array to 2D
        self.sim_data['theta'] = theta #p.reshape(
        #     a=theta,
        #     newshape=(self._numtimesteps, self._numlocations),
        #     order='F'
        # )

        return self


    def calculateIAM(self, a1: float = 0.000884 , a2: float = 0.00005369, a3: float = 0):
        """ Calculates the IAM angle modifier from incidence angle. Formula and default values are from: 
        [1]	GAFUROV, Tokhir, Julio USAOLA, and Milan PRODANOVIC. Modelling of concentrating
        solar power plant for power system reliability studies [online]. IET Renewable Power
        Generation. 2015, 9(2), 120-130. Available from: 10.1049/iet-rpg.2013.0377. 

        Args:
            a1 (float, optional): IAM-modifier1. Defaults to 0.000884.
            a2 (float, optional): IAM-modifier1. Defaults to 0.00005369.
            a3 (float, optional): IAM-modifier1. Defaults to 0.

        Returns:
            [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for sim_data['IAM'][timeserie_iter, location_iter]
        """

        # check for input data availability
        assert 'theta' in list(self.sim_data.keys())


        _theta = self.sim_data['theta'] # deg

        #calculate with formula for IAM: IAM = 1 + 'sum over i'  a_i * (theta^i / cos (theta))
        
        # a_i, theta in deg
        _IAM = 1 - a1 * np.power(_theta, 1) / np.cos(np.deg2rad(_theta)) \
            - a2 * np.power(_theta, 2) / np.cos(np.deg2rad(_theta)) \
            - a3 * np.power(_theta, 3) / np.cos(np.deg2rad(_theta))

        #replace nan with zero
        self.sim_data['IAM'] = np.nan_to_num(_IAM, nan = 0)

        return self


    def calculateShadowLosses(self, SF_density: float = 0.383, method: str = 'wagner2011'):
        """Estimates shadow losses from solar field density and solar altitude
        
        Args:
            SF_density (float, optional): [description]. Defaults to 0.383.
            method(str, optional): [choose from 'wagner2011' and 'gafurov2015']

        Returns:
            [type]: [description]

        References:
        [1]	WAGNER, Michael J. and Paul GILMAN. Technical Manual for the SAM Physical Trough Model, 2011.
        [2]	GAFUROV, Tokhir, Julio USAOLA, and Milan PRODANOVIC. Modelling of concentrating
        solar power plant for power system reliability studies [online]. IET Renewable Power
        Generation. 2015, 9(2), 120-130. Available from: 10.1049/iet-rpg.2013.0377.
        """
        method = method.lower()
        assert 'solar_altitude_angle_degree' in self.sim_data.keys()
        
        if method =='wagner2011':
            # equation 2.38 from [1]	WAGNER, Michael J. and Paul GILMAN. Technical Manual for the SAM Physical Trough Model, 2011.
            # keep in mind, that cos(zenith) is replaced by sin(solar altitude angle)
            # value output is limited to 0.5 ... 1
            self.sim_data['eta_shdw'] = np.minimum(np.abs(np.sin(np.deg2rad(self.sim_data['solar_altitude_angle_degree']))) / SF_density, 1)

        elif method == 'gafurov2015':
            warning('The method gafurov2015 for shadow losses is not fully implemented!')
            self.sim_data['eta_shdw'] = np.sin(np.deg2rad(self.sim_data['solar_altitude_angle_degree'])) / ( SF_density * np.cos(np.deg2rad(self.sim_data['theta'])) )

        return self


    def calculateDegradationLosses(self, efficencyDropPerYear = 0, lifetime = 40):
            if efficencyDropPerYear == 0:
                self.sim_data['eta_degradation'] = 1
            else:
                self.sim_data['eta_degradation'] = (1 - (1-efficencyDropPerYear)**(lifetime+1))/(1-(1-efficencyDropPerYear)) / lifetime


    def calculateWindspeedLosses(self, max_windspeed_threshold: float = 14):
        """ If windspeed is above threshold, the efficency is set to zero.

        Args:
            max_windspeed_threshold (float, optional): [description]. Defaults to 9999.

        Returns:
            [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for sim_data['eta_wind'][timeserie_iter, location_iter] as np.narray
        """

        assert "surface_wind_speed" in self.sim_data.keys()

        self.sim_data['eta_wind'] = np.less_equal(self.sim_data['surface_wind_speed'], max_windspeed_threshold).astype(int)

        return self


    def calculateHeattoHTF(self, eta_ptr_max: float = 0.742, eta_cleaness: float = 1):
        """Calculates the heat from Collector to heat transfer fluid. The result is before the heat losses of the HTF.

        Args: 
            eta_ptr_max (float, optional): [Value for optical efficency of trough mirror and absorber]. Defaults to 0.742.
            eta_cleaness (float, optional): Cleannes factor of the solar receiver (mirrors)
            A_aperture_sf (int, optional): [Size of the solar field in m^2]. Defaults to 909060.

        Returns:
            [type]: [description]
        """

        # units
        # eta_ptr_max: 1
        # costheta: 1
        # IAM: 1
        # eta_shdw: 1
        # A_aperture_sf: m^2
        # direct_normal_irradiance: W/m^2

        self.eta_ptr_max = eta_ptr_max
        self.eta_cleaness = eta_cleaness
        

        self.sim_data['HeattoHTF_W'] = self.eta_ptr_max \
                                        * self.eta_cleaness \
                                        * np.cos(np.deg2rad(self.sim_data['theta'])) \
                                        * self.sim_data['IAM'] \
                                        * self.sim_data['eta_shdw'] \
                                        * self.sim_data['eta_wind'] \
                                        * self.sim_data['eta_degradation'] \
                                        * self.placements['aperture_area_m2'].values \
                                        * self.sim_data['direct_normal_irradiance']

        self.sim_data['P_DNI'] = self.placements['aperture_area_m2'].values * self.sim_data['direct_normal_irradiance']

        self.sim_data['P_DNI_eta_opt'] = self.placements['aperture_area_m2'].values * self.sim_data['direct_normal_irradiance'] * eta_ptr_max


        return self


    def applyHTFHeatLossModel(self, calculationmethod: str = 'gafurov2013', params: dict = {}):
        """Calculate the heat losses of the HTF and determines the Heat output of the solar field

        Args:
            calculationmethod (str, optional): [calculation method for heat losses. Choose from 'zero' or 'gafurov2013']. Defaults to 'zero'.
            params (dict, optional): [Parameters for the heat models as dict. For 'gafurov2013' use relHeatLosses and ratedFieldOutputHeat_W]. Defaults to {}.

        Raises:
            error: [description]

        Returns:
            [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for sim_data['HeattoPlant_W'][timeserie_iter, location_iter] as np.narray
        """
        #check for input
        assert 'HeattoHTF_W' in self.sim_data.keys()
        calculationmethod = calculationmethod.lower()



        if calculationmethod == "zero":
           _losses = np.zeros_like(self.sim_data['HeattoHTF_W'], dtype = float)
        
        elif calculationmethod == "gafurov2013":
            
            #check for input
            assert 'relHeatLosses' in params.keys()
            assert 'ratedFieldOutputHeat_W' in params.keys()

            _losses = np.ones_like(self.sim_data['HeattoHTF_W'], dtype = float) * params['relHeatLosses'] * params['ratedFieldOutputHeat_W']
            self.sim_data['HeattoPlant_W'] = self.sim_data['HeattoHTF_W'] - _losses
            self.sim_data['HeatLosses_W'] = _losses

        elif calculationmethod == "dersch2018":

            #check for input
            assert 'b' in params.keys()
            assert 'relTMplant' in params.keys()
            assert 'maxHTFTemperature' in params.keys()
            assert 'JITaccelerate' in params.keys()
            assert 'minHTFTemperature' in params.keys()
            assert 'inletHTFTemperature' in params.keys()
            assert params['b'].shape == (5,)
            assert 'add_losses_coefficient' in params.keys()

            assert 'IAM' in self.sim_data.keys()
            assert 'theta' in self.sim_data.keys()
            assert 'surface_air_temperature' in self.sim_data.keys()
            assert 'HeattoHTF_W' in self.sim_data.keys()
            assert 'direct_normal_irradiance' in self.sim_data.keys()


            #set up arrays
            #timedelta in seconds
            deltat = self.time_index[1] - self.time_index[0]
            deltat = deltat.total_seconds() #seconds
            #temperature
            _temperature = np.empty_like(self.sim_data['HeattoHTF_W'])
            _temperature[0, :] = self.sim_data['surface_air_temperature'][0,:] + 100 #initial temperature
            #losses
            _losses = np.zeros_like(self.sim_data['HeattoHTF_W'])
            #K = cos(theta) * IAM
            _K = np.cos(np.deg2rad(self.sim_data['theta'])) * self.sim_data['IAM']
            # heating power needed:
            _P_heating = np.zeros_like(self.sim_data['HeattoHTF_W'])
            # heat to Plant
            _HeattoPlant = np.zeros_like(self.sim_data['HeattoHTF_W'])
    
            if params['discretizationmethod'] == 'euler explicit':
                def simulation(
                    HeattoHTF: np.ndarray, temperature: np.ndarray, ambient_temperature: np.ndarray,
                    losses: np.ndarray, K: np.ndarray, DNI: np.ndarray, A: np.ndarray, b:np.ndarray, relTMplant: float,
                    deltat: float, maxHTFTemperature: float, minHTFTemperature: float, inletHTFTemperature: float,
                    P_heating: np.ndarray, HeattoPlant: np.ndarray, add_losses_coefficient: float):
                    """[Transient simulation of the HTF fluid temperature. Calculate losses by empiric formulation as in Greenius]

                    Args:
                        HeattoHTF (np.ndarray): [description]
                        temperature (np.ndarray): [description]
                        ambient_temperature (np.ndarray): [description]
                        losses (np.ndarray): [description]
                        K (np.ndarray): [description]
                        DNI (np.ndarray): [description]
                        A (float): [description]
                        b (np.ndarray): [description]
                        TM_plant (float): [description]
                        deltat (float): [description]
                        maxHTFTemperature (float): [description]

                    Returns:
                        [type]: [description]
                    """
                    maxHTFmeanTemperature = (maxHTFTemperature + inletHTFTemperature)/2
                    minHTFmeanTemperature = minHTFTemperature # because plant is not at operation, there is no normal outlet temperature

                    for i in range(0, temperature.shape[0]-1):
                        
                        # 1) Delta T
                        # deltaT = (T_out - T in) / - T_amb  ...  (in, out from the solar field view, defined in Greenius)
                        deltaT = temperature[i, :] - ambient_temperature[i, :]
                        
                        # 2) Losses
                        # loss formula from greenius
                        losses[i, :] = K[i, :] * b[0] * deltaT * A * DNI[i, :] + A * ( b[1] * deltaT**1 + b[2] * deltaT**2 + b[3] * deltaT**3 + b[4] * deltaT**4) 
                        # additional loss factor:
                        losses[i, :] = losses[i, :] + add_losses_coefficient * deltaT * A
                        #losses[i, :] = losses[i, :] * heatlossfactor + heatlossconstant

                        # calculate temperature from energy balance around all thermal masses which need to be heated up (see sam manual)
                        temperature[i+1, :] = temperature[i, :] + (HeattoHTF[i+1, :] - losses[i, :]) * deltat / (relTMplant * A)
                        # The heat to plant is neglected in this formualtion. For this will be accouintet in the next lines
                        
                        # The temperature should always stay between max temperature and min temperature
                        # 1) When the Temperatrue is at the operation point, excess heat will be used in the plant --> Normal operation
                        # 2) When the temperature is at the minimum point, missing heat will be provided from electric heating

                        # 1) Operation Point
                        # maximal temperature is achieved, when outlet temperature is at max temperature.
                        temperature[i+1, :] = np.minimum(temperature[i+1, :], maxHTFmeanTemperature)
                        # Heat to plant equals Heat input minus losses, when the plant is in operation mode (temperature = max temperature)
                        
                        HeattoPlant[i, :] = (HeattoHTF[i, :] - losses[i, :]) * (temperature[i, :] == maxHTFmeanTemperature)
                        # In the first time step with max temp, not all energy can be used in the plant, as some energy was used for heating the htf.
                        # The htf heating losses are substracted for the first time step 
                        is_first_max_heat = np.logical_and(temperature[i-1, :] != maxHTFmeanTemperature, temperature[i, :] == maxHTFmeanTemperature)
                        heat_flux_htf_heatup_last_timestep = (temperature[i, :] - temperature[i-1, :]) * relTMplant * A / deltat

                        # manipulate HeattoPlant, so that heat flux into htf is substracted from HeattoPlant
                        HeattoPlant[i, :] = np.maximum(HeattoPlant[i, :] - is_first_max_heat * heat_flux_htf_heatup_last_timestep, 0)
                        # because of discretization uncertainties, HeattoPlant can get negativ. So this will be prevented here. Looks odd, but is true

                        # 2) Freeze protection
                        # when temperature is below minimal temperature, there will be an electrical heating for the HTF so that:
                        # 2.1) temperatere is locked at min temp
                        temperature[i+1, :] = np.maximum(temperature[i+1, :], minHTFmeanTemperature)
                        # 2.2) there are parasitic losses for heating
                        # Heating equals the heat losses - the heat input from the field, if the plant is in freez protection mode(temperature = min temperature)
                        P_heating[i, :] = np.maximum((losses[i, :] - HeattoHTF[i, :]) * (temperature[i, :] == minHTFmeanTemperature), 0)
                        # because of discretization uncertainties, P_heating can get negativ. So this will be prevented here. Looks odd, but is true

                    return temperature, losses, P_heating, HeattoPlant

            elif params['discretizationmethod'] == 'euler implicit':
                def simulation(
                    HeattoHTF: np.ndarray, temperature: np.ndarray, ambient_temperature: np.ndarray,
                    losses: np.ndarray, K: np.ndarray, DNI: np.ndarray, A: np.ndarray, b:np.ndarray, relTMplant: float,
                    deltat: float, maxHTFTemperature: float, minHTFTemperature: float, inletHTFTemperature: float,
                    P_heating: np.ndarray, HeattoPlant: np.ndarray, add_losses_coefficient: float):
                    """[Transient simulation of the HTF fluid temperature. Calculate losses by empiric formulation as in Greenius]

                    Args:
                        HeattoHTF (np.ndarray): [description]
                        temperature (np.ndarray): [description]
                        ambient_temperature (np.ndarray): [description]
                        losses (np.ndarray): [description]
                        K (np.ndarray): [description]
                        DNI (np.ndarray): [description]
                        A (float): [description]
                        b (np.ndarray): [description]
                        TM_plant (float): [description]
                        deltat (float): [description]
                        maxHTFTemperature (float): [description]

                    Returns:
                        [type]: [description]
                    """
                    maxHTFmeanTemperature = (maxHTFTemperature + inletHTFTemperature)/2
                    minHTFmeanTemperature = minHTFTemperature # because plant is not at operation, there is no normal outlet temperature

                    for i in range(0, temperature.shape[0]-1):
                        
                        # 1) Delta T
                        # deltaT = (T_out - T in) / - T_amb  ...  (in, out from the solar field view, defined in Greenius)
                        
                        
                        # 2) Losses
                        # get polinomial factors of eueler forward discretized equation:
                        # iterate trough locations
                        for i_loc in range (0, HeattoHTF.shape[1]):
                            p0 = - HeattoHTF[i+1, i_loc] - (relTMplant * A) / deltat * (temperature[i, i_loc] - ambient_temperature[i, i_loc])
                            p1 = (relTMplant * A) / deltat + K[i+1, i_loc] * b[0] * A * DNI[i+1, i_loc] + A * b[1] + add_losses_coefficient * A
                            p2 = A * b[2]
                            p3 = A * b[3]
                            p4 = A * b[4]

                            # solve polynomial for zeros:
                            solutions_T_star = np.roots(np.array([p4,p3,p2,p1,p0]).astype(np.complex128))
                            solutions_T = solutions_T_star - ambient_temperature[i+1, i_loc]
                            
                            # filter out solutions which are real and between -200 and 1000 °C
                            solutions_T_filtered = []
                            for solution in solutions_T:
                                if solution.imag < 1E-10 and solution.imag > -1E-10:
                                    if solution.real > -200 and solution.real <1000:
                                        solutions_T_filtered.append(solution)
                            
                            if len(solutions_T_filtered)!=1:
                                print(solutions_T)
                                print('Severe error: Multiple or no zeros for temperature calculation were found!!')
                                print('Timestep: ' + str(i) + ', Location: ' + str(i_loc))
                                #error('Severe error: Multiple or no zeros for temperature calculation were found!!')


                            #set temperature
                            temperature[i+1, i_loc] = solutions_T_filtered[0].real


                        # Set temperature limits
                        temperature[i+1, :] = np.minimum(temperature[i+1, :], maxHTFmeanTemperature)
                        temperature[i+1, :] = np.maximum(temperature[i+1, :], minHTFmeanTemperature)

                        # calculate delta T
                        deltaT = temperature[i+1, :] - ambient_temperature[i+1, :]

                        # loss formula from greenius
                        losses[i+1, :] = K[i, :] * b[0] * deltaT * A * DNI[i, :] + A * ( b[1] * deltaT**1 + b[2] * deltaT**2 + b[3] * deltaT**3 + b[4] * deltaT**4) 
                        # additional loss factor:
                        losses[i+1, :] = losses[i+1, :] + add_losses_coefficient * deltaT * A
                        #losses[i, :] = losses[i, :] * heatlossfactor + heatlossconstant

                        # calculate temperature from energy balance around all thermal masses which need to be heated up (see sam manual)
                        ########
                        # temperature[i+1, :] = temperature[i, :] + (HeattoHTF[i+1, :] - losses[i, :]) * deltat / (relTMplant * A)
                        # 
                        # The heat to plant is neglected in this formualtion. For this will be accouintet in the next lines
                        
                        # The temperature should always stay between max temperature and min temperature
                        # 1) When the Temperatrue is at the operation point, excess heat will be used in the plant --> Normal operation
                        # 2) When the temperature is at the minimum point, missing heat will be provided from electric heating

                        # 1) Operation Point
                        # maximal temperature is achieved, when outlet temperature is at max temperature.
                        temperature[i+1, :] = np.minimum(temperature[i+1, :], maxHTFmeanTemperature)
                        # Heat to plant equals Heat input minus losses, when the plant is in operation mode (temperature = max temperature)
                        HeattoPlant[i, :] = np.maximum((HeattoHTF[i, :] - losses[i, :]) * (temperature[i, :] == maxHTFmeanTemperature), 0)
                        # because of discretization uncertainties, HeattoPlant can get negativ. So this will be prevented here. Looks odd, but is true

                        # 2) Freeze protection
                        # when temperature is below minimal temperature, there will be an electrical heating for the HTF so that:
                        # 2.1) temperatere is locked at min temp
                        temperature[i+1, :] = np.maximum(temperature[i+1, :], minHTFmeanTemperature)
                        # 2.2) there are parasitic losses for heating
                        # Heating equals the heat losses - the heat input from the field, if the plant is in freez protection mode(temperature = min temperature)
                        P_heating[i, :] = np.maximum((losses[i, :] - HeattoHTF[i, :]) * (temperature[i, :] == minHTFmeanTemperature), 0)
                        # because of discretization uncertainties, P_heating can get negativ. So this will be prevented here. Looks odd, but is true

                    return temperature, losses, P_heating, HeattoPlant
                
            
            # NUMBA JIT the simulation
            tic = time.time()
            simulation_jitted = jit(nopython=True)(simulation) 


            # Run the simulation
            if not params['JITaccelerate']:
                _temperature, _losses, _P_heating, _HeattoPlant \
                    = simulation(
                        HeattoHTF=self.sim_data['HeattoHTF_W'],
                        temperature=_temperature,
                        ambient_temperature=self.sim_data['surface_air_temperature'],
                        losses=_losses,
                        K=_K,
                        DNI=self.sim_data['direct_normal_irradiance'],
                        A=self.placements['aperture_area_m2'].values,
                        b=params['b'],
                        relTMplant=params['relTMplant'],
                        deltat=deltat,
                        maxHTFTemperature=params['maxHTFTemperature'],
                        minHTFTemperature=params['minHTFTemperature'],
                        inletHTFTemperature=params['inletHTFTemperature'],
                        P_heating = _P_heating,
                        HeattoPlant = _HeattoPlant,
                        add_losses_coefficient = params['add_losses_coefficient']
                    )

            else:
                _temperature, _losses, _P_heating, _HeattoPlant \
                    = simulation_jitted(
                        HeattoHTF=self.sim_data['HeattoHTF_W'],
                        temperature=_temperature,
                        ambient_temperature=self.sim_data['surface_air_temperature'],
                        losses=_losses,
                        K=_K,
                        DNI=self.sim_data['direct_normal_irradiance'],
                        A=self.placements['aperture_area_m2'].values,
                        b=params['b'],
                        relTMplant=params['relTMplant'],
                        deltat=deltat,
                        maxHTFTemperature=params['maxHTFTemperature'],
                        minHTFTemperature=params['minHTFTemperature'],
                        inletHTFTemperature=params['inletHTFTemperature'],
                        P_heating = _P_heating,
                        HeattoPlant = _HeattoPlant,
                        add_losses_coefficient = params['add_losses_coefficient']
                    )

            toc = time.time()
            print('Core simulation time: ' + str(toc-tic) + 's.')



            #store data
            self.sim_data['HTF_mean_temperature_C'] = _temperature
            self.sim_data['Heat_Losses_W'] = _losses
            self.sim_data['P_heating_W'] = _P_heating
            self.sim_data['HeattoPlant_W'] = _HeattoPlant

        elif calculationmethod == "exact":
            warn('Wrong calculation for heat losses of heat transfer fluid selected. Losses will be set to zero.')
            _losses = np.zeros_like(self.sim_data['HeattoHTF_W'], dtype = float)
            self.sim_data['HeattoPlant_W'] = self.sim_data['HeattoHTF_W'] - _losses
            self.sim_data['HeatLosses_W'] = _losses

        else:
            warn('Wrong calculation for heat losses of heat transfer fluid selected. Losses will be set to zero.')
            _losses = np.zeros_like(self.sim_data['HeattoHTF_W'], dtype = float)
            self.sim_data['HeattoPlant_W'] = self.sim_data['HeattoHTF_W'] - _losses
            self.sim_data['HeatLosses_W'] = _losses
        
        

        return self


    def calculateParasitics(self, calculationmethod: str = 'gafurov2013', params: dict = {}):
        '''Calculating the parasitic losses of the plant

        Parameters
        ----------
        calculationmethod : str, optional
            [description], by default 'gafurov2013'
        params : dict, optional
            For calculationmethod gafurov013:
                I_DNI_nom: DNI for design point estimation in W/m^2
                PL_plant_fix: Fixed plant losses in % of design point power output of the plant
                PL_sf_track: Fixed solar field losses in % of design point power output of the field
                PL_sf_pumping: Solar field pumping losses in % of design point power output
                PL_plant_other: Plant Pumping losses in % of design point power output

        '''
        if calculationmethod == 'gafurov2013':
            assert 'I_DNI_nom' in params.keys()
            assert 'PL_plant_fix' in params.keys()
            assert 'PL_sf_track' in params.keys()
            assert 'PL_sf_pumping' in params.keys()
            assert 'PL_plant_pumping' in params.keys()
            assert 'PL_plant_other' in params.keys()


            ##### CALCULATION

            #### estimate design parameters:

        
            # Q_sf,des is the design point power output of the plant
            # P_pb_des is the design point power output of the plant
            nominal_sf_efficiency = np.max(self.eta_ptr_max \
                                            * self.eta_cleaness \
                                            * np.cos(np.deg2rad(self.sim_data['theta'])) \
                                            * self.sim_data['IAM'] \
                                            * self.sim_data['eta_shdw'])
            nominal_efficiency_power_block = 0.3774 # 37.74% efficency of the power block at nominal power, from gafurov2013
            nominal_receiver_heat_losses = 0.06 # 6% losses nominal heat losses, from gafurov2013
            SM = 2

            Q_sf_des = nominal_sf_efficiency * self.placements['aperture_area_m2'].values * params['I_DNI_nom'] * (1-nominal_receiver_heat_losses) #W
            P_pb_des = Q_sf_des * nominal_efficiency_power_block / SM

            # PL_csp,fix
            PL_plant_fix = params['PL_plant_fix'] * P_pb_des

            #PL_sf_track 
            PL_sf_track = params['PL_sf_track'] * P_pb_des * (self.sim_data['solar_zenith_degree'] < 90)

            #PL_sf_night = self.sim_data['P_heating_W']

            #PL_sf_pumping
            PL_sf_pumping = params['PL_sf_pumping'] * Q_sf_des * np.power(self.sim_data['HeattoPlant_W'] / Q_sf_des, 3)

            #PL_plant_pumping
            PL_plant_pumping = params['PL_plant_pumping'] * Q_sf_des

            #PL_plant_other
            PL_plant_other = params['PL_plant_other'] * P_pb_des

            self.sim_data['PL_sf_track'] = PL_sf_track
            self.sim_data['PL_sf_pumping'] = PL_sf_pumping

            self.sim_data['Parasitics_solarfield_W'] = PL_sf_track + self.sim_data['P_heating_W'] + PL_sf_pumping
            self.sim_data['Parasitics_plant_W'] = (PL_plant_fix + PL_plant_pumping + PL_plant_other) * np.ones_like(self.sim_data['Parasitics_solarfield_W'])
            self.sim_data['Parasitics_total_W'] = self.sim_data['Parasitics_solarfield_W'] + self.sim_data['Parasitics_plant_W']

        return self

    
    def calculateEconomics_SolarField(self, WACC: float = 8, lifetime: float = 25,  calculationmethod: str = 'franzmann2021', params: dict = {}):
        '''Calculating the cost for internal heat from CSP

        Parameters
        ----------
        WACC : float, optional
            [description], by default 8
        lifetime : float, optional
            [description], by default 25
        calculationmethod : str, optional
            [description], by default 'franzmann2021'
        params : dict, optional
            [description], by default {}

        Returns
        -------
        [type]
            [description]
        '''
        #calculate from percent to abs value
        if WACC > 0.5:
            WACC = WACC / 100
        # Calculate annuity factor from WACC and lifetime like in Heuser
        self.sim_data['annuity'] = (WACC * (1 + WACC)**lifetime) / ((1+WACC)**lifetime - 1)
        
        dt = (self._time_index_[1] - self._time_index_[0]) / pd.Timedelta(hours=1)
        # calculate the average annual heat production
        self.sim_data['annualHeat_Wh'] = self.sim_data['HeattoPlant_W'].mean(axis=0)  * ((self._time_index_[-1] - self._time_index_[0]) / pd.Timedelta(hours=1))
        
        if calculationmethod == 'franzmann2021':
            #assert 'CAPEX_solar_field_USD_per_m^2_aperture' in params.keys(), "'CAPEX_solar_field_USD_per_m^2_aperture' needs to be in params"
            #assert 'CAPEX_land_USD_per_m^2_land' in params.keys(), "'CAPEX_land_USD_per_m^2_land' needs to be in params"
            #assert 'fixOPEX_%_CAPEX_per_a' in params.keys(), "'_CAPEX_per_a' needs to be in params"
            #assert 'indirect_cost_%_CAPEX' in params.keys(), "'indirect_cost_USD' needs to be in params"
            
            self.sim_data['CAPEX_SF_USD'] = (self.placements['aperture_area_m2'] * params['CAPEX_solar_field_USD_per_m^2_aperture'] \
                        + self.placements['land_area_m2'] * params['CAPEX_land_USD_per_m^2_land']) \
                        * (1 + params['CAPEX_indirect_cost_%_CAPEX'] / 100)
                 
            
        elif False:
            pass
        
        #calcualte annual Costs
        self.sim_data['Capex_SF_USD_per_a'] = self.sim_data['CAPEX_SF_USD'] * self.sim_data['annuity']
        self.sim_data['opexFix_SF_USD_per_a'] = self.sim_data['CAPEX_SF_USD'] * params['OPEX_%_CAPEX'] / 100
        
        #calculate opex
        if 'Parasitics_solarfield_W' in self.sim_data.keys():
            dt = (self._time_index_[1] - self._time_index_[0]) / pd.Timedelta(hours=1) 
            self.sim_data['opexVar_SF_USD_per_a'] = self.sim_data['Parasitics_solarfield_W'].sum() / 1000 * dt * params['electricity_price_USD_per_kWh']
        else:
            self.sim_data['opexVar_SF_USD_per_a'] = 0

        #calculate annual Totex
        self.sim_data['Totex_SF_USD_per_a'] = self.sim_data['Capex_SF_USD_per_a'] + self.sim_data['opexFix_SF_USD_per_a'] + self.sim_data['opexVar_SF_USD_per_a']

        #Cost relative to Heat
        self.sim_data['LCO_Heat_SF_USD_per_Wh'] = self.sim_data['Totex_SF_USD_per_a'] / self.sim_data['annualHeat_Wh']

        

        return self
    
    
    def Sizing(self, I_DNI_nom):
        
        assert 'Totex_SF_USD_per_a' in self.sim_data.keys()
        assert 'HeattoPlant_W' in self.sim_data.keys()
        assert 'aperture_area_m2' in self.placements.keys()
        assert 'direct_normal_irradiance' in self.sim_data.keys()
        assert 'theta' in self.sim_data.keys()
        assert 'IAM' in self.sim_data.keys()
        assert 'eta_shdw' in self.sim_data.keys()
        
        #estimate parameters
        nominal_sf_efficiency = np.max(self.eta_ptr_max \
                                        * self.eta_cleaness \
                                        * np.cos(np.deg2rad(self.sim_data['theta'])) \
                                        * self.sim_data['IAM'] \
                                        * self.sim_data['eta_shdw'])
        #nominal_efficiency_power_block = 0.3774 # 37.74% efficency of the power block at nominal power, from gafurov2013
        nominal_receiver_heat_losses = 0.06 # 6% losses nominal heat losses, from gafurov2013
        
        I_DNI_nom = np.minimum(self.sim_data['direct_normal_irradiance'].max(axis=0)*0.8, I_DNI_nom)

        Q_sf_des = nominal_sf_efficiency * self.placements['aperture_area_m2'].values * I_DNI_nom * (1-nominal_receiver_heat_losses) #W
        
        self.sm = np.array([2.1])#[1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 5.5, 6, 6.5, 7])
        self.tes = np.array([7.5])#[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ,20])
        
        self.dimensions = [self.sim_data['HeattoPlant_W'].shape[0], self.sim_data['HeattoPlant_W'].shape[1], len(self.sm), len(self.tes)]
        
        #create raw Heat Plant        
        #dimensions: [time(hours), placements, SM]
        HeatfromField_W_3D = np.tensordot(
            self.sim_data['HeattoPlant_W'],
            np.ones(shape=(self.dimensions[2])),
            axes=0
        )
        
        # annualHeatfromField_Wh_3D = np.tensordot(
        #     HeatfromField_W_3D.sum(axis=0),
        #     np.ones(shape=(dimensions[3])),
        #     axes=0
        # )
        
        
        #calculate max possible heat useage 
        
        #calculate max possible powerplant consumption
        #Consumption = Q_sf_des * 1 / SM
        #dimensions: [time(hours), placements, SM]
        Powerplant_consumption_max_W_3D = np.tensordot(
            np.tensordot(
                np.ones(self.dimensions[0]),
                Q_sf_des, axes=0
            ),
            1 / self.sm,
            axes=0
        )
        
        #heat for direct useage in power plant
        #sum over heat for direct usage (min from available from field and max produceable from plant)
        #dimensions: [time(hours), placements, SM]
        directHeatUsage_Wh_3D_ts = np.minimum(HeatfromField_W_3D, Powerplant_consumption_max_W_3D)
        
        #dimensions: [placements, SM]
        directHeatUsage_Wh_2D = directHeatUsage_Wh_3D_ts.sum(axis=0)
        
        #dimensions: [ placements, SM, TES]
        directHeatUsage_Wh_3D = np.tensordot(
            directHeatUsage_Wh_2D,
            np.ones(self.dimensions[3]),
            axes=0
        )
        del directHeatUsage_Wh_2D
        
        #calculate the heat that mus be stored (because plant smaller than field)
        #dimensions: [time(hours), placements, SM]
        HeattoStorage_W_3D = np.maximum(HeatfromField_W_3D - Powerplant_consumption_max_W_3D, 0)
        
        #dimensions: [days, hoursperyear]
        aggregate_by_day = np.eye(365).repeat(24, axis=1)
        self.aggregate_by_day = aggregate_by_day
        
        #aggregate the stored heat for each day
        #dimensions: [time(days), placements, SM]
        dailyHeatStorable_Wh_3D = np.einsum('ij,jkl', aggregate_by_day, HeattoStorage_W_3D)
        del HeattoStorage_W_3D
        
        #convert to 4D
        #dimensions: [time(days), placements, SM, TES]
        dailyHeatStorable_Wh_4D = np.tensordot(
            dailyHeatStorable_Wh_3D,
            np.ones(self.dimensions[3]),
            axes=0,
        )
        del dailyHeatStorable_Wh_3D
        
        #limit storage by thermal storage size
        #dimensions: [time(days), placements, SM, TES]
        storage_Size_Wh_4D = np.tensordot(
            np.tensordot(
                np.tensordot(
                    np.ones(aggregate_by_day.shape[0]), #days
                    Q_sf_des,
                    axes=0
                ),
                1/self.sm,#np.ones(self.dimensions[2]),
                axes=0
            ),
            self.tes,
            axes=0
        )
        
        #actual stored heat daily
        #dimensions: [time(days), placements, SM, TES]
        dailyHeatStored_Wh_4D = np.minimum(dailyHeatStorable_Wh_4D, storage_Size_Wh_4D)
        del dailyHeatStorable_Wh_4D, storage_Size_Wh_4D
        
        # daily used direct heat:
        #dimensions: [time(days), placements, SM]
        dailyHeatDirect_Wh_3D = np.einsum('ij,jkl', aggregate_by_day, directHeatUsage_Wh_3D_ts)
        del directHeatUsage_Wh_3D_ts
        
        #dimensions: [time(days), placements, SM, TES]
        dailyHeatDirect_Wh_4D = np.tensordot(
            dailyHeatDirect_Wh_3D,
            np.ones(self.dimensions[3]),
            axes=0,
        )
        del dailyHeatDirect_Wh_3D
        
        #max Heat input to PowerPlant
        #dimensions: [placements, SM, TES]
        maxDailyHeatPlant_W_3D = np.tensordot(
            Powerplant_consumption_max_W_3D[0,:,:],
            np.ones(self.dimensions[3]),
            axes=0,
        ) * 24 #h/day
        #dimensions: [time(days), placements, SM, TES]
        maxDailyHeatPlant_W_4D = np.tensordot(
            np.ones(aggregate_by_day.shape[0]),
            maxDailyHeatPlant_W_3D,
            axes=0
        )
        del maxDailyHeatPlant_W_3D
        
        #limit daily output
        #dimensions: [time(days), placements, SM, TES]
        dailyHeatOutput_Wh_4D = np.minimum((dailyHeatDirect_Wh_4D + dailyHeatStored_Wh_4D), maxDailyHeatPlant_W_4D)
        del maxDailyHeatPlant_W_4D, dailyHeatDirect_Wh_4D
        
        
        #dimensions: [placements, SM, TES]
        annualHeat_Wh_3D = dailyHeatOutput_Wh_4D.sum(axis=0)
                
        self.sim_data['annualHeat_Wh_3D'] = annualHeat_Wh_3D #dimensions: [placements, SM, TES]
        self.sim_data['dailyHeatOutput_Wh_4D'] = dailyHeatOutput_Wh_4D #dimensions: [time(days), placements, SM, TES]
        self.placements['I_DNI_nom_W_per_m2'] = I_DNI_nom
        self.placements['Q_sf_des_W'] = Q_sf_des
        # #stored heat per year
        # #dimensions: [placements, SM, TES]
        # annualHeatStored_Wh_3D = dailyHeatStored_Wh_4D.sum(axis=0)
        
        # #total heat storable per year
        # #dimensions: [placements, SM, TES]
        # annualHeatstoreable_Wh_3D = annualHeatStored_Wh_3D + directHeatUsage_Wh_3D
        
        # #max Heat input to PowerPlant
        # #dimensions: [placements, SM, TES]
        # maxAnnualHeatPlant_W_3D = np.tensordot(
        #     Powerplant_consumption_max_W_3D[0,:,:],
        #     np.ones(dimensions[3]),
        #     axes=0,
        # ) * aggregate_by_day.shape[1] # hours per  year
        
        # #total heat useable per year
        # #dimensions: [placements, SM, TES]
        # annualHeat_Wh_3D = np.minimum(annualHeatstoreable_Wh_3D, maxAnnualHeatPlant_W_3D)
        return self
    
    
    def calculateEconomics_Plant_Storage(self, params):
        
        assert 'CAPEX_plant_cost_USD_per_kW' in params.keys()
        assert 'CAPEX_storage_cost_USD_per_kWh' in params.keys()
        #Cost estiamtations for plant and storage:
        #dimensions: [SM, TES]
        sm_2D = np.tile(self.sm, (self.dimensions[3], 1)).T
        tes_2D = np.tile(self.tes, (self.dimensions[2], 1))
        
        #CAPEX_Plant_Storage per Solar field size
        #dimensions: [SM, TES]
        CAPEX_USD_per_kW_SF_2D = (params['CAPEX_plant_cost_USD_per_kW'] / sm_2D + params['CAPEX_storage_cost_USD_per_kWh'] * tes_2D / sm_2D) * (1 + params['CAPEX_indirect_cost_%_CAPEX']/100)
        
        #yearly cost of storage and plant
        #dimensions: [SM, TES]
        CAPEX_USD_per_a_kW_SF_2D = CAPEX_USD_per_kW_SF_2D * self.sim_data['annuity']
        varOPEX_USD_per_a_kW_SF_2D = CAPEX_USD_per_kW_SF_2D * params['OPEX_%_CAPEX'] / 100
        fixOPEX_USD_per_a_kW_SF_2D = 0
        
        #dimensions: [SM, TES]
        TOTEX_Plant_storage_USD_per_a_kw_SF_2D = CAPEX_USD_per_a_kW_SF_2D + varOPEX_USD_per_a_kW_SF_2D + fixOPEX_USD_per_a_kW_SF_2D
        del CAPEX_USD_per_a_kW_SF_2D, varOPEX_USD_per_a_kW_SF_2D, fixOPEX_USD_per_a_kW_SF_2D
        
        #TOTEX plant and storage per year (abolute)
        #Q_sf_des / 1000 * speccosts_USD_per_kw_sf_2D
        #dimensions: [placements, SM, TES]
        TOTEX_Plant_storage_USD_per_a_3D = np.tensordot(
            self.placements['Q_sf_des_W'] / 1000,
            TOTEX_Plant_storage_USD_per_a_kw_SF_2D,
            axes=0
            )
        
        #costs of field:
        #dimensions: [placements, SM, TES]
        TOTEX_SF_USD_per_a_3D = np.tensordot(
            np.tensordot(
                self.sim_data['Totex_SF_USD_per_a'],
                np.ones(self.dimensions[2]),
                axes=0,
            ),
            np.ones(self.dimensions[3]),
            axes=0,
        )
        #dimensions: [placements, SM, TES]
        TOTEX_CSP_USD_per_a_3D = TOTEX_SF_USD_per_a_3D + TOTEX_Plant_storage_USD_per_a_3D
            

        # cost per heat
        #dimensions: [placements, SM, TES]
        LCO_Heat_USD_per_Wh = TOTEX_CSP_USD_per_a_3D / self.sim_data['annualHeat_Wh_3D']
        
        #find minimum:
        #dimensions: [placements]
        sm_opt = []
        tes_opt = []
        #loop placemnts(I did not find a function wich gives the argmin along two axes (1,2))
        for i in range(0, self.dimensions[1]):
            temp = LCO_Heat_USD_per_Wh[i,:,:]
            #find minimum index
            min_tuple = np.unravel_index(np.argmin(temp, axis=None), temp.shape)
            #append
            sm_opt.append(self.sm[min_tuple[0]])
            tes_opt.append(self.tes[min_tuple[1]])
                

        self.placements['SM_opt'] = sm_opt
        self.placements['TES_opt_h'] = tes_opt
        self.sim_data['LCO_Heat_USD_per_Wh'] = LCO_Heat_USD_per_Wh
        self.sim_data['TOTEX_CSP_USD_per_a_3D'] = TOTEX_CSP_USD_per_a_3D
        #dbg
        self.sim_data['LCO_Heat_USD_per_Wh'] = LCO_Heat_USD_per_Wh
        self.sim_data['TOTEX_SF_USD_per_a_3D'] = TOTEX_SF_USD_per_a_3D
        self.sim_data['TOTEX_Plant_storage_USD_per_a_3D'] = TOTEX_Plant_storage_USD_per_a_3D



        
        return self
            
        
    def simulate_electric_output(self):
        #Calculate average LCOE
        
        
        #dimensions: [time(days), placements, SM, TES]
        self.sim_data['dailyHeatOutput_Wh_4D']
        
        
        #dimensions: [time(days), placements, SM, TES]
        Q_plant_des_Wh_per_day_4D = np.tensordot(
            np.tensordot(
                np.tensordot(
                    np.ones(self.aggregate_by_day.shape[0]),   #days
                    self.placements['Q_sf_des_W'],             #placements 
                    axes=0                                  
                ),
                1 / self.sm,                                   #SM
                axes=0
            ),
            np.ones(self.dimensions[3]),                       #TES
            axes=0
        ) * 24 #h/dey
        
        #dimensions: [time(days), placements, SM, TES]
        rel_load_plant_4D = self.sim_data['dailyHeatOutput_Wh_4D'] / Q_plant_des_Wh_per_day_4D
        
        #Gafurov2015: 
        # rel_efficiency [%] = 54.92 + 112.73 * rel - 104.63 * rel^2 + 37.05 * rel^3
        efficiency_daily_averaged_1_4D = (0.5492 \
                + 1.1273 * rel_load_plant_4D \
                - 1.0463 * rel_load_plant_4D**2 \
                + 0.3705 * rel_load_plant_4D**3)
        
        self.sim_data['dailyPowerOutput_Wh_4D'] = efficiency_daily_averaged_1_4D * self.sim_data['dailyHeatOutput_Wh_4D']
        
        LCOE_USD_per_Wh = self.sim_data['TOTEX_CSP_USD_per_a_3D'] / self.sim_data['annualHeat_Wh_3D']
        
        #find minimum:
        #dimensions: [placements]
        sm_opt = []
        tes_opt = []
        #loop placemnts(I did not find a function wich gives the argmin along two axes (1,2))
        for i in range(0, self.dimensions[1]):
            temp = LCOE_USD_per_Wh[i,:,:]
            #find minimum index
            min_tuple = np.unravel_index(np.argmin(temp, axis=None), temp.shape)
            #append
            sm_opt.append(self.sm[min_tuple[0]])
            tes_opt.append(self.tes[min_tuple[1]])
        
        return self


    ### Try to increase speed of PV-Lib by dropping one loop. Aparently, multiple locations are not supported by PV-Lib. If there are performance
    ### issues, try again. So keep this in mind here 
    #
    def calculateSolarPositionfaster(self):
        """
        DOES NOT WORK PV LIP DOES NOT SUPPORT MULTIPLE LOCATIONS
        calculates the solar position in terms of hour angle and declination from time series and location series of the current object
    
        Returns:
            [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for sim_data['hour_angle'][timeserie_iter, location_iter] and 
            sim_data['declination_angle'][timeserie_iter, location_iter]]
        """
        
        assert 'lat' in self.placements.columns
        assert 'lon' in self.placements.columns
        assert 'elev' in self.placements.columns
        
        # Shape Input data to 1D
        _time = pd.DatetimeIndex(np.tile(self.time_index.values, self._numlocations))
        #_dayoftheyear = pd.DataFrame(np.tile(self.time_index.day_of_year.values, self._numlocations))
        _latitute = pd.DataFrame(self.placements['lat'].values.repeat(self._numtimesteps))
        _longitude = pd.DataFrame(self.placements['lon'].values.repeat(self._numtimesteps))
        _elevation = pd.DataFrame(self.placements['elev'].values.repeat(self._numtimesteps))

        _solarpos = pvlib.solarposition.spa_python(
            _time.values.squeeze(),
            latitude=_latitute.values.squeeze(),
            longitude=_longitude.values.squeeze(),
            #altitude=_elevation.values.squeeze(),
        )
        
        self.sim_data['solar_zenith_degree_fast'] = np.reshape(
            a=_solarpos['apparent_zenith'].values,
            newshape=(self._numlocations, self._numtimesteps)
        ).T
        
        #calculate aoi
        truetracking_angles = pvlib.tracking.singleaxis(
            apparent_zenith=_solarpos['apparent_zenith'],
            apparent_azimuth=_solarpos['azimuth'],
            axis_tilt=0,
            axis_azimuth=180,
            max_angle=90,
            backtrack=False,  # for true-tracking
            gcr=0.5)  # irrelevant for true-tracking

        _aoi_northsouth = np.nan_to_num(truetracking_angles['aoi'].values)
        self.sim_data['aoi_northsouth_fast'] = np.reshape(
            a=_aoi_northsouth,
            newshape=(self._numlocations, self._numtimesteps)
        ).T

        #calculate aoi
        truetracking_angles = pvlib.tracking.singleaxis(
            apparent_zenith=_solarpos['apparent_zenith'],
            apparent_azimuth=_solarpos['azimuth'],
            axis_tilt=0,
            axis_azimuth=90,
            max_angle=180,
            backtrack=False,  # for true-tracking
            gcr=0.5)  # irrelevant for true-tracking
        
        _aoi_eastwest = np.nan_to_num(truetracking_angles['aoi'].values)
        self.sim_data['aoi_eastwest_fast'] = np.reshape(
            a=_aoi_eastwest,
            newshape=(self._numlocations, self._numtimesteps)
        ).T

        #from [1]	KALOGIROU, Soteris A. Environmental Characteristics. In: Soteris Kalogirou, ed. Solar energy engineering. Processes and systems. Waltham, Mass: Academic Press, 2014, pp. 51-123.
        # fromula 2.12
        
        self.sim_data['solar_altitude_angle_degree_fast'] = \
            np.rad2deg(np.arcsin(np.cos(np.deg2rad(self.sim_data['solar_zenith_degree_fast']))))

        return self