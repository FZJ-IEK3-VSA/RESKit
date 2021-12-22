from logging import warn, warning

from reskit.csp.data.database_loader import load_dataset
from reskit.solar.workflows.solar_workflow_manager import SolarWorkflowManager
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
        #self.module = None
        self.sim_data_daily = dict()

        self.check_placements()
        
    def check_placements(self):
        assert hasattr(self, 'placements')
        assert isinstance(self.placements, pd.DataFrame)
        assert 'lat' in self.placements.columns or 'latitude' in self.placements.columns
        assert 'lon' in self.placements.columns or 'longitude' in self.placements.columns
        assert 'land_area_m2' in self.placements.columns \
            or 'aperture_area_m2' in self.placements.columns \
            or 'area' in self.placements.columns \
            or 'area_m2' in self.placements.columns
    
    def loadPTRdata(self, datasetname:str):
        '''loads the dataset with the name datasetname.

        Parameters
        ----------
        datasetname : str
            [description]
        '''
        self.ptr_data = load_dataset(datasetname=datasetname)
        # make list from coefficients from regression
        self.ptr_data['b'] = np.array([
            self.ptr_data['b0'],
            self.ptr_data['b1'],
            self.ptr_data['b2'],
            self.ptr_data['b3'],
            self.ptr_data['b4'],
        ])
        self.placements['datasetname'] = datasetname
        return self.ptr_data
        
    def determine_area(self):
        '''determines the land area, aperture area from given placement dataframe.
        If only 'area' is given, it will be assumed as land area.
        '''
        assert hasattr(self, 'ptr_data')
        assert 'SF_density_total' in self.ptr_data.index
        columns = self.placements.columns
        
        #if only area in placements:
        if 'area' in  columns and not 'aperture_area_m2' in columns and not 'land_area_m2' in columns:
            warn('Key "area" is assumed to be the land area. Abort if wrong!')
            self.placements['land_area_m2'] = self.placements['area']
            self.placements.drop('area', axis=1)
            self.placements['aperture_area_m2'] = self.placements['land_area_m2'] * self.ptr_data['SF_density_total']

        if 'area_m2' in  columns and not 'aperture_area_m2' in columns and not 'land_area_m2' in columns:
            warn('Key "area" is assumed to be the land area. Abort if wrong!')
            self.placements['land_area_m2'] = self.placements['area_m2']
            self.placements.drop('area_m2', axis=1)
            self.placements['aperture_area_m2'] = self.placements['land_area_m2'] * self.ptr_data['SF_density_total']

        #only aperture_area_m2 in placements
        elif 'aperture_area_m2' in columns and not 'land_area_m2' in columns:
            self.placements['aperture_area_m2'] = self.placements['land_area_m2'] * self.ptr_data['SF_density_total']
        
        #only land_area_m2 in placements
        elif 'land_area_m2' in columns and not 'aperture_area_m2' in columns:
            self.placements['land_area_m2'] = self.placements['aperture_area_m2'] / self.ptr_data['SF_density_total']                          
    
    def get_timesteps(self):
        self._numtimesteps = self.time_index.shape[0]
        self._numlocations = self.placements.shape[0]
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

    def apply_azimuth(self):
        """
        Applies the azimuth angle for each placement. Three options:
        'northsouth':
            azimuth is always 180° (N-S)
        'eastwerst':
            azimuth is always 90° (E-W)
        'song2013':
            azimuth determined by lat: if |lat|<46: N-S, else: E-W
        
        params:
        orientation: str
            see above
        """
        orientation = self.ptr_data['orientation']
        accepted_orientation = [
            'northsouth',
            'eastwest',
            'song2013'
        ]
        assert orientation in accepted_orientation
        
        if orientation == 'northsouth':
            self.placements['azimuth'] = 180
        elif orientation == 'eastwest':
            self.placements['azimuth'] = 90
        elif orientation == 'song2013':
            
            def apply_song2013(lat):
                ''' apply song 2013: if latitude is between -46° and +46°, use northsouth orientation, else eastwest orientation'''
                
                if lat < 46.06 and lat > -46.06:
                    # northsouth
                    return 180
                else:
                    # eastwest
                    return 90
            
            self.placements['azimuth'] = self.placements['lat'].apply(apply_song2013)
    
    def apply_capacity_sf(self):
        assert 'aperture_area_m2' in self.placements.keys()
        assert 'direct_normal_irradiance' in self.sim_data.keys()
        assert 'theta' in self.sim_data.keys()
        assert 'IAM' in self.sim_data.keys()
        assert 'eta_shdw' in self.sim_data.keys()
        
        #estimate parameters
        nominal_sf_efficiency = np.max(self.ptr_data['eta_ptr_max'] \
                                        * self.ptr_data['eta_cleaness'] \
                                        * np.cos(np.deg2rad(self.sim_data['theta'])) \
                                        * self.sim_data['IAM'] \
                                        * self.sim_data['eta_shdw'])
        #nominal_efficiency_power_block = 0.3774 # 37.74% efficency of the power block at nominal power, from gafurov2013
        nominal_receiver_heat_losses = 0.06 # 6% losses nominal heat losses, from gafurov2013
        
        I_DNI_nom = np.percentile(self.sim_data['direct_normal_irradiance'], 90, axis=0) #np.minimum(self.sim_data['direct_normal_irradiance'].max(axis=0), self.ptr_data['I_DNI_nom'])

        Q_sf_des = nominal_sf_efficiency * self.placements['aperture_area_m2'].values * I_DNI_nom * (1-nominal_receiver_heat_losses) #W
        
        self.placements['capacity_sf_W_th'] = Q_sf_des
        self.placements['I_DNI_nom_W_per_m2'] = I_DNI_nom
    
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
        assert 'azimuth' in self.placements.columns
        assert hasattr(self, 'time_index')
        assert self.ptr_data['SF_density_direct'] < 1

        #set up empty array
        self.sim_data['solar_zenith_degree'] = np.empty(shape=(self._numtimesteps, self._numlocations))
        self.sim_data['theta'] = np.empty(shape=(self._numtimesteps, self._numlocations))
        self.sim_data['tracking_angle'] = np.empty(shape=(self._numtimesteps, self._numlocations))



        # iterate trough all location
        for location_iter, row in enumerate(self.placements[['lon', 'lat', 'elev', 'azimuth']].itertuples()):

            #calculate the solar position
            _solarpos = \
                pvlib.solarposition.get_solarposition(
                    time=self.time_index,
                    latitude=row.lat,
                    longitude=row.lon,
                    altitude=row.elev,
                    #pressure=self.sim_data["surface_pressure"][:, location_iter], #TODO: insert here
                    #temperature=self.sim_data["surface_air_temperature"][:, location_iter], #TODO: insert here
                    method='nrel_numba'
                )


            self.sim_data['solar_zenith_degree'][:, location_iter] = _solarpos['apparent_zenith'].values
            #self.sim_data['solar_azimuth_degree'][:, location_iter] = _solarpos['azimuth'].values

            #calculate aoi
            truetracking_angles = pvlib.tracking.singleaxis(
                apparent_zenith=_solarpos['apparent_zenith'],
                apparent_azimuth=_solarpos['azimuth'],
                axis_tilt=0,
                axis_azimuth=row.azimuth,
                max_angle=90,
                backtrack=False,  # for true-tracking
                gcr=self.ptr_data['SF_density_direct'])  # irrelevant for true-tracking

            self.sim_data['theta'][:, location_iter] = np.nan_to_num(truetracking_angles['aoi'].values)
            self.sim_data['tracking_angle'][:, location_iter] = np.nan_to_num(truetracking_angles['tracker_theta'].values)

            #from [1]	KALOGIROU, Soteris A. Environmental Characteristics. In: Soteris Kalogirou, ed. Solar energy engineering. Processes and systems. Waltham, Mass: Academic Press, 2014, pp. 51-123.
            # fromula 2.12
        
        self.sim_data['solar_altitude_angle_degree'] = np.rad2deg(np.arcsin(np.cos(np.deg2rad(self.sim_data['solar_zenith_degree']))))

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
        self.sim_data['IAM'] = np.maximum(np.nan_to_num(_IAM, nan = 0), 0)

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
        
        
        if method =='wagner2011':
            # equation 2.38 from [1]	WAGNER, Michael J. and Paul GILMAN. Technical Manual for the SAM Physical Trough Model, 2011.
            # keep in mind, that cos(zenith) is replaced by sin(solar altitude angle)
            # value output is limited to 0... 1
            #self.sim_data['eta_shdw'] = np.minimum(np.abs(np.sin(np.deg2rad(self.sim_data['solar_altitude_angle_degree']))) / SF_density, 1)
            assert 'tracking_angle' in self.sim_data.keys()
            assert 'solar_zenith_degree' in self.sim_data.keys()
            
            self.sim_data['eta_shdw'] = np.minimum(np.abs(np.cos(np.deg2rad(self.sim_data['tracking_angle']))) / SF_density, 1) #TODO
            self.sim_data['eta_shdw'][self.sim_data['solar_zenith_degree']>90] = 0

        elif method == 'gafurov2015':
            warning('The method gafurov2015 for shadow losses is not fully implemented!')
            assert 'solar_altitude_angle_degree' in self.sim_data.keys()
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


    def calculateHeattoHTF(self, eta_ptr_max: float = 0.742, eta_cleaness: float = 1, eta_other: float = 0.99):
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

        self.sim_data['HeattoHTF_W'] = eta_ptr_max \
                                        * eta_cleaness \
                                        * eta_other \
                                        * np.cos(np.deg2rad(self.sim_data['theta'])) \
                                        * self.sim_data['IAM'] \
                                        * self.sim_data['eta_shdw'] \
                                        * self.sim_data['eta_wind'] \
                                        * self.sim_data['eta_degradation'] \
                                        * self.placements['aperture_area_m2'].values \
                                        * self.sim_data['direct_normal_irradiance']

        #self.sim_data['P_DNI'] = self.placements['aperture_area_m2'].values * self.sim_data['direct_normal_irradiance']

        #self.sim_data['P_DNI_eta_opt'] = self.placements['aperture_area_m2'].values * self.sim_data['direct_normal_irradiance'] * eta_ptr_max


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
            _temperature = np.zeros_like(self.sim_data['HeattoHTF_W'])
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
            self.sim_data['Heat_Losses_W'] = _losses

        else:
            warn('Wrong calculation for heat losses of heat transfer fluid selected. Losses will be set to zero.')
            _losses = np.zeros_like(self.sim_data['HeattoHTF_W'], dtype = float)
            self.sim_data['HeattoPlant_W'] = self.sim_data['HeattoHTF_W'] - _losses
            self.sim_data['Heat_Losses_W'] = _losses
        
        

        return self


    def calculateParasitics(self, calculationmethod: str = 'gafurov2013', params: dict = {}):
        '''Calculating the parasitic losses of the plant

        Parameters
        ----------
        calculationmethod : str, optional
            [description], by default 'gafurov2013'
        params : dict, optional
            For calculationmethod gafurov013:
                PL_plant_fix: Fixed plant losses in % of design point power output of the plant
                PL_sf_track: Fixed solar field losses in % of design point power output of the field
                PL_sf_pumping: Solar field pumping losses in % of design point power output
                PL_plant_other: Plant Pumping losses in % of design point power output

        '''
         #### estimate design parameters:

        
        # Q_sf,des is the design point power output of the solar field
        # P_pb_des is the design point power output of the plant
        nominal_efficiency_power_block = self.ptr_data['eta_powerplant_1']
        SM = 2

        Q_sf_des = self.placements['capacity_sf_W_th'].values #W
        P_pb_des = Q_sf_des * nominal_efficiency_power_block / SM
        
        P_pb = self.sim_data['HeattoPlant_W'] * nominal_efficiency_power_block
        
        
        if calculationmethod == 'gafurov2013':
            assert 'PL_plant_fix' in params.keys()
            assert 'PL_sf_track' in params.keys()
            assert 'PL_sf_pumping' in params.keys()
            assert 'PL_plant_pumping' in params.keys()
            assert 'PL_plant_other' in params.keys()


            ##### CALCULATION

            # PL_csp,fix
            PL_plant_fix = params['PL_plant_fix'] * P_pb

            #PL_sf_track 
            PL_sf_track = params['PL_sf_track'] * P_pb_des * (self.sim_data['solar_zenith_degree'] < 90)

            #PL_sf_pumping
            PL_sf_pumping = params['PL_sf_pumping'] * Q_sf_des * np.power(self.sim_data['HeattoPlant_W'] / Q_sf_des , 3)
            #PL_plant_pumping
            PL_plant_pumping = params['PL_plant_pumping'] * self.sim_data['HeattoPlant_W']

            #PL_plant_other
            PL_plant_other = params['PL_plant_other'] * P_pb

            #self.sim_data['PL_sf_track'] = PL_sf_track
            #self.sim_data['PL_sf_pumping'] = PL_sf_pumping

            self.sim_data['Parasitics_solarfield_W_el'] = PL_sf_track + PL_sf_pumping #+ self.sim_data['P_heating_W']#issue #13
            self.sim_data['Parasitics_plant_W_el'] = (PL_plant_fix + PL_plant_pumping + PL_plant_other)
            
        
        elif calculationmethod == 'dersch2018':
            
            params['PL_sf_fixed_W_per_m^2_ap'] = 1 * 1.486
            params['PL_sf_pumping_W_per_m^2_ap'] = 8.3
            
            assert 'PL_sf_fixed_W_per_m^2_ap' in params.keys()
            assert 'PL_sf_pumping_W_per_m^2_ap' in params.keys()
            
            PL_sf_track = params['PL_sf_fixed_W_per_m^2_ap'] * self.placements['aperture_area_m2'].values * (self.sim_data['HeattoHTF_W']> 0)
            PL_sf_pumping = params['PL_sf_pumping_W_per_m^2_ap'] * self.placements['aperture_area_m2'].values * \
                    np.power(self.sim_data['HeattoPlant_W'] / Q_sf_des, 2) # * 830/self.placements['I_DNI_nom_W_per_m2'].values), 2) #used for valiadaton, as 830 as DNI_des is used in reference data
            
            self.sim_data['Parasitics_solarfield_W_el'] = PL_sf_track + PL_sf_pumping # + self.sim_data['P_heating_W']#issue #13
            
            #Plant from Gaforov
            PL_plant_fix = params['PL_plant_fix'] * P_pb
            PL_plant_pumping = params['PL_plant_pumping'] * self.sim_data['HeattoPlant_W']
            PL_plant_other = params['PL_plant_other'] * P_pb
            self.sim_data['Parasitics_plant_W_el'] = (PL_plant_fix + PL_plant_pumping + PL_plant_other)
            
        else:
            raise ValueError('calculationmethod for parasitic losses not known. Use "gafurov2013" or "dersch2018"')
        
        self.sim_data['Parasitics_W_el'] = self.sim_data['Parasitics_solarfield_W_el'] + self.sim_data['Parasitics_plant_W_el']
        self.placements['Parasitics_solarfield_Wh_el_per_a'] = self.sim_data['Parasitics_solarfield_W_el'].sum(axis=0)
        self.placements['Parasitics_plant_Wh_el_per_a'] = self.sim_data['Parasitics_plant_W_el'].sum(axis=0)
        
        assert not np.isnan(self.sim_data['Parasitics_solarfield_W_el']).any()
        assert not np.isnan(self.sim_data['Parasitics_plant_W_el']).any()
            
        return self

    def calculateCapacityFactors(self):
        
        if not 'capacity_sf_W_th' in self.placements.columns:
            self.apply_capacity_sf() 
        assert 'HeattoPlant_W' in self.sim_data.keys()
        self.sim_data['capacity_factor_sf'] = self.sim_data['HeattoPlant_W'] / np.tile(self.placements['capacity_sf_W_th'], (8760,1))
        self.sim_data_daily['capacity_factor_plant'] = self.sim_data_daily['Power_net_total_per_day_Wh'] / (self.placements['power_plant_capacity_W_el'].values*24)
        
    def calculateEconomics_SolarField(self, WACC: float = 8, lifetime: float = 25,  calculationmethod: str = 'franzmann2021', params: dict = {}):
        '''Calculating the cost for internal heat from CSP. 
        CAPEX: Contains solar field cost, land cost, indirect cost for solar field
        OPEX: fixOPEX ist 2%/a of CAPEX and varCAPEX is electricity demand for solar field pumping  

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
        if WACC > 1:
            WACC = WACC / 100
        # Calculate annuity factor from WACC and lifetime like in Heuser
        self.sim_data['annuity'] = (WACC * (1 + WACC)**lifetime) / ((1+WACC)**lifetime - 1)
        
        dt = (self._time_index_[1] - self._time_index_[0]) / pd.Timedelta(hours=1)
        # calculate the average annual heat production
        self.placements['annualHeatfromSF_Wh'] = self.sim_data['HeattoPlant_W'].mean(axis=0)  * ((self._time_index_[-1] - self._time_index_[0]) / pd.Timedelta(hours=1))
        
        if calculationmethod == 'franzmann2021':
            #assert 'CAPEX_solar_field_EUR_per_m^2_aperture' in params.keys(), "'CAPEX_solar_field_EUR_per_m^2_aperture' needs to be in params"
            #assert 'CAPEX_land_EUR_per_m^2_land' in params.keys(), "'CAPEX_land_EUR_per_m^2_land' needs to be in params"
            #assert 'fixOPEX_%_CAPEX_per_a' in params.keys(), "'_CAPEX_per_a' needs to be in params"
            #assert 'indirect_cost_%_CAPEX' in params.keys(), "'indirect_cost_EUR' needs to be in params"
            
            self.placements['CAPEX_SF_EUR'] = (self.placements['aperture_area_m2'] * params['CAPEX_solar_field_EUR_per_m^2_aperture'] \
                        + self.placements['land_area_m2'] * params['CAPEX_land_EUR_per_m^2_land']) \
                        * (1 + params['CAPEX_indirect_cost_%_CAPEX'] / 100)
                 
            
        elif False:
            pass
        
        #calcualte annual Costs
        Capex_SF_EUR_per_a = self.placements['CAPEX_SF_EUR'] * self.sim_data['annuity']
        opexFix_SF_EUR_per_a = self.placements['CAPEX_SF_EUR'] * params['OPEX_%_CAPEX'] / 100
        
        #calculate opex
        dt = (self._time_index_[1] - self._time_index_[0]) / pd.Timedelta(hours=1) 
        opexVar_SF_EUR_per_a = self.sim_data['Parasitics_solarfield_W_el'].sum(axis=0) / 1000 * dt * params['electricity_price_EUR_per_kWh']


        #calculate annual Totex
        self.placements['Totex_SF_EUR_per_a'] = Capex_SF_EUR_per_a + opexFix_SF_EUR_per_a + opexVar_SF_EUR_per_a

        #Cost relative to Heat
        self.placements['LCO_Heat_SF_EURct_per_kWh'] = self.placements['Totex_SF_EUR_per_a'] / self.placements['annualHeatfromSF_Wh'] * 1E2 * 1E3 # EUR/Wh to EURct/kWh

        

        return self
    
    
    def optimize_plant_size(self, onlynightuse=True, fullvariation=False, debug_vars=False):
        '''returns the optimal pLant configuration for each placement by finding the lowest expeected LCOE: sm_opt, tes opt

        Parameters
        ----------
        onlynightuse : bool, optional
            allheat has to be stored, i order to be used contrary to PV, by default True
        fullvariation : bool, optional
            for plotting purpose, full variation can be set to true to caclucate variation over more values, by default False
        '''
        
        #check inputs
        assert not np.isnan(self.placements['capacity_sf_W_th']).any()
        assert not np.isnan(self.sim_data['HeattoPlant_W']).any()

        # for developers: use full variations
        if fullvariation:
            self.sm = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
            self.tes = np.array([5, 6, 9, 12, 15, 18])
        else:
            self.sm = np.array([1.5, 2, 2.5, 3, 3.5])
            self.tes = np.array([9, 12, 15])
        
        #make lsit with all sizing combinations
        sizing_tuples = []
        for sm in self.sm:
            for tes in self.tes:
                sizing_tuples.append((sm, tes))
        
        assert len(sizing_tuples) == len(self.sm)*len(self.tes)
        
        
        #loop all sizing combinations
        #dimensions: [time(days), placements, SM, TES]
        if debug_vars:
            dailyHeatOutput_Wh_4D = np.nan*np.ones(shape=(365, len(self.placements), len(self.sm), len(self.tes))) #TODO: remove
            TOTEX_EUR_per_a_3D = np.nan*np.ones(shape=(len(self.placements), len(self.sm), len(self.tes))) #TODO: remove
            Power_output_plant_net_Wh_per_a_3D = np.nan*np.ones(shape=(len(self.placements), len(self.sm), len(self.tes))) #TODO: remove
        LCOE_EURct_per_kWh_el_3D = np.nan*np.ones(shape=(len(self.placements), len(self.sm), len(self.tes)))
        for size in sizing_tuples:
            sm = size[0]
            tes = size[1]
            
            #get index
            i_sm = np.where(self.sm == sm)[0][0]
            i_tes = np.where(self.tes == tes)[0][0]
            
            
            ##################################
            ### 1) get opt thermal output  ###
            ##################################
            
            #get thermal size of the power plant
            Heatflux_powerplant_des_input_W_th = self.placements['capacity_sf_W_th'] / sm
            Heat_Storage_des_Wh_th = Heatflux_powerplant_des_input_W_th * tes
            
            #get direct power to plant
            if onlynightuse:
                Heatflux_direct_powerplant_W_th = 0 
            else:
                Heatflux_direct_powerplant_W_th = np.minimum(self.sim_data['HeattoPlant_W'], Heatflux_powerplant_des_input_W_th)
            
            #get heat to be stored
            Heatflux_to_storage_W_th = np.maximum(self.sim_data['HeattoPlant_W'] - Heatflux_direct_powerplant_W_th,0)
            
            
            #aggregate to daily
            aggregate_by_day = np.eye(365).repeat(24, axis=1)
            self.aggregate_by_day = aggregate_by_day
            
            #aggregate the stored heat for each day
            Heat_to_storage_daily_Wh_th = np.einsum('ij,jk', aggregate_by_day, Heatflux_to_storage_W_th)
            Heat_heating_sf_daily_Wh_th = np.einsum('ij,jk', aggregate_by_day, self.sim_data['P_heating_W']) #13
            
            if onlynightuse:
                Heat_direct_powerplant_daily_Wh_th = 0
            else:
                Heat_direct_powerplant_daily_Wh_th = np.einsum('ij,jk', aggregate_by_day, Heatflux_direct_powerplant_W_th)
            del Heatflux_direct_powerplant_W_th, Heatflux_to_storage_W_th
            
            #limit dayliy heat output from storage by storage size
            Heat_stored_daily_Wh_th = np.minimum(Heat_to_storage_daily_Wh_th, Heat_Storage_des_Wh_th)
            del Heat_to_storage_daily_Wh_th
            
            #calculate heat which is unstored
            Heat_unstored_daily_Wh_th = np.maximum(Heat_stored_daily_Wh_th * self.ptr_data['storage_efficiency_1'] - Heat_heating_sf_daily_Wh_th, 0) #13
            P_backup_heating_daily_Wh_el = np.maximum(Heat_heating_sf_daily_Wh_th - Heat_stored_daily_Wh_th * self.ptr_data['storage_efficiency_1'],0)#13
            del Heat_stored_daily_Wh_th
            
            #max heat processable by power plant
            if onlynightuse:
                operationalhours_per_day = np.einsum('ij,jk', aggregate_by_day, (self.sim_data['solar_zenith_degree']>90))
            else:
                operationalhours_per_day = 24
            Heat_max_des_powerplant_daily_Wh_th = Heatflux_powerplant_des_input_W_th.values * operationalhours_per_day # h/day
            
            #calculate actually used heat
            Heat_total_used_daily_Wh_th = np.minimum((Heat_direct_powerplant_daily_Wh_th + Heat_unstored_daily_Wh_th), Heat_max_des_powerplant_daily_Wh_th)
            del Heat_direct_powerplant_daily_Wh_th, Heat_unstored_daily_Wh_th, Heat_max_des_powerplant_daily_Wh_th
            
            #remember that one
            if debug_vars:
                dailyHeatOutput_Wh_4D[:,:,i_sm, i_tes] = Heat_total_used_daily_Wh_th

            ##################################
            ### 2) get cost                ###
            ##################################
            
            TOTEX_EUR_per_a = self._get_totex_from_self(sm_manipulation = sm, tes_manipulation = tes)
            
            if debug_vars:           
                TOTEX_EUR_per_a_3D[:,i_sm, i_tes] = TOTEX_EUR_per_a
            
            ##################################
            ### 3) Electric Output         ###
            ##################################
            
            #calcualte average rel load of the plant
            rel_load_plant_daily_1 = Heat_total_used_daily_Wh_th / (Heatflux_powerplant_des_input_W_th.values * operationalhours_per_day)
        
            # calcualte plant efficiency
            efficiency_daily_1 = self._get_plant_efficiency(rel_load_plant=rel_load_plant_daily_1, eta_nom=self.ptr_data['eta_powerplant_1'])
            del rel_load_plant_daily_1
            
            # gross power output
            Power_output_plant_gross_daily_Wh = Heat_total_used_daily_Wh_th * efficiency_daily_1
            del efficiency_daily_1
            #plant parasitics
            Parasitics_plant_daily_Wh_el = np.einsum('ij,jk', aggregate_by_day, self.sim_data['Parasitics_W_el']) *1#h #13
            Parasitics_plant_daily_Wh_el += P_backup_heating_daily_Wh_el #13
            #net power output
            Power_output_plant_net_daily_Wh = np.maximum(Power_output_plant_gross_daily_Wh - Parasitics_plant_daily_Wh_el,0)
            del Parasitics_plant_daily_Wh_el, Power_output_plant_gross_daily_Wh
            
            #sum up
            Power_output_plant_net_Wh_per_a = Power_output_plant_net_daily_Wh.sum(axis=0).squeeze()
            
            if debug_vars:
                Power_output_plant_net_Wh_per_a_3D[:,i_sm, i_tes] = Power_output_plant_net_Wh_per_a
            
            ##################################
            ### 4) LCOE                    ###
            ##################################
            
            #with np.seterr(divide='ignore', invalid='ignore') #TODO: zero devision
            LCOE_EUR_per_kWh_el = np.nan_to_num(TOTEX_EUR_per_a.values / Power_output_plant_net_Wh_per_a * 1E5, nan=0) #EUR/Wh --> EURct/kWh
            
            LCOE_EURct_per_kWh_el_3D[:,i_sm, i_tes] = LCOE_EUR_per_kWh_el #TODO: remove this, only dbg
            
        #check if all is written
        assert not np.isnan(LCOE_EURct_per_kWh_el_3D).any()
        
        #container for opt variables
        if not hasattr(self, 'opt_data'):
            self.opt_data = {}
        if debug_vars:    
            self.opt_data['dailyHeatOutput_Wh_4D_new'] = dailyHeatOutput_Wh_4D
            self.opt_data['TOTEX_EUR_per_a_3D_new'] = TOTEX_EUR_per_a_3D
            self.opt_data['LCOE_EURct_per_kWh_el_3D_new'] = LCOE_EURct_per_kWh_el_3D
        
        ##################################
        ### 5) opt                     ###
        ##################################

        #find minimum:
        #dimensions: [placements]
        sm_opt = []
        tes_opt = []
        #loop placemnts(I did not find a function wich gives the argmin along two axes (1,2))
        for i in range(0, len(self.placements)):
            temp = LCOE_EURct_per_kWh_el_3D[i,:,:]
            #find minimum index
            sm_opt_index, tes_opt_index = np.unravel_index(np.argmin(temp, axis=None), temp.shape)
            #append
            sm_opt.append(self.sm[sm_opt_index])
            tes_opt.append(self.tes[tes_opt_index])        

        # set minimum values to placement df
        self.placements['sm_opt'] = sm_opt
        self.placements['tes_opt'] = tes_opt
        self.placements['storage_capacity_kWh_th'] = self.placements['capacity_sf_W_th'] / sm_opt * tes_opt / 1000
        self.placements['power_plant_capacity_W_el'] = self.placements['capacity_sf_W_th'] / sm_opt * self.ptr_data['eta_powerplant_1']

    def optimize_heat_output_4D(self):
        '''calculates the heat usage for different type of plant configuration: SM and TES
            Calcualtes the following variables:
                Annual heat: Total heat, that can be used annually for the plant configuration
                Direct heat Usage: Heat, that must be directly processed throgh the plant (storage size limitations)
                Stored heat: Heat, that is stored and can be used daily at any given time


        Returns
        -------
        [type]
            [description]
        '''
        assert 'Totex_SF_EUR_per_a' in self.placements.columns
        assert 'HeattoPlant_W' in self.sim_data.keys()
        
        #estimate parameters
        # nominal_sf_efficiency = np.max(self.ptr_data['eta_ptr_max'] \
        #                                 * self.ptr_data['eta_cleaness'] \
        #                                 * np.cos(np.deg2rad(self.sim_data['theta'])) \
        #                                 * self.sim_data['IAM'] \
        #                                 * self.sim_data['eta_shdw'])
        # #nominal_efficiency_power_block = 0.3774 # 37.74% efficency of the power block at nominal power, from gafurov2013
        # nominal_receiver_heat_losses = 0.06 # 6% losses nominal heat losses, from gafurov2013

        Q_sf_des = self.placements['capacity_sf_W_th'] 

        self.sm = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])#, 5.5, 6, 6.5, 7]) #np.array([2.1])
        self.tes = np.array([0, 3, 6, 9, 12, 15, 18])#([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])#, 16, 17, 18, 19 ,20]) #np.array([7.5])
        
        if not hasattr(self, 'opt_data'):
            self.opt_data = {} #container for opt variables
        self.opt_data['dimensions'] = [
            self.sim_data['HeattoPlant_W'].shape[0], #time
            self.sim_data['HeattoPlant_W'].shape[1], #placements
            len(self.sm), # SM variation
            len(self.tes) # TES variation
        ]
        
        #create raw Heat Plant        
        #dimensions: [time(hours), placements, SM]
        HeatfromField_W_3D = np.tensordot(
            self.sim_data['HeattoPlant_W'],
            np.ones(shape=(self.opt_data['dimensions'][2])),
            axes=0
        )      
        
        #calculate max possible heat useage 
        
        #calculate max possible powerplant consumption
        #Consumption = Q_sf_des * 1 / SM
        #dimensions: [time(hours), placements, SM]
        Powerplant_consumption_max_W_3D = np.tensordot(
            np.tensordot(
                np.ones(self.opt_data['dimensions'][0]),
                Q_sf_des, axes=0
            ),
            1 / self.sm,
            axes=0
        )
        
        #heat for direct useage in power plant
        #sum over heat for direct usage (min from available from field and max produceable from plant)
        #dimensions: [time(hours), placements, SM]
        directHeatUsage_Wh_3D_ts = np.minimum(HeatfromField_W_3D, Powerplant_consumption_max_W_3D)
        Powerplant_consumption_max_W_2D = Powerplant_consumption_max_W_3D[0,:,:]
        del Powerplant_consumption_max_W_3D
        
        #dimensions: [placements, SM]
        directHeatUsage_Wh_2D = directHeatUsage_Wh_3D_ts.sum(axis=0)
        
        #dimensions: [ placements, SM, TES]
        directHeatUsage_Wh_3D = np.tensordot(
            directHeatUsage_Wh_2D,
            np.ones(self.opt_data['dimensions'][3]),
            axes=0
        )
        del directHeatUsage_Wh_2D
        
        #calculate the heat that must be stored (because plant smaller than field)
        #dimensions: [time(hours), placements, SM]
        HeattoStorage_W_3D = np.maximum(HeatfromField_W_3D - directHeatUsage_Wh_3D_ts,0) #np.maximum(HeatfromField_W_3D - Powerplant_consumption_max_W_3D, 0)
        del HeatfromField_W_3D

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
            np.ones(self.opt_data['dimensions'][3]),
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
                1/self.sm,#np.ones(self.opt_data['dimensions'][2]),
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
        self.opt_data['directHeatUsage_Wh_3D_ts'] = directHeatUsage_Wh_3D_ts
        del directHeatUsage_Wh_3D_ts
        
        #dimensions: [time(days), placements, SM, TES]
        dailyHeatDirect_Wh_4D = np.tensordot(
            dailyHeatDirect_Wh_3D,
            np.ones(self.opt_data['dimensions'][3]),
            axes=0,
        )
        del dailyHeatDirect_Wh_3D
        
        #max Heat input to PowerPlant
        #dimensions: [placements, SM, TES]
        maxDailyHeatPlant_W_3D = np.tensordot(
            Powerplant_consumption_max_W_2D, #Powerplant_consumption_max_W_3D[0,:,:],
            np.ones(self.opt_data['dimensions'][3]),
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
        dailyHeatOutput_Wh_4D = np.minimum((dailyHeatDirect_Wh_4D + dailyHeatStored_Wh_4D * self.ptr_data['storage_efficiency_1']), maxDailyHeatPlant_W_4D)
        del maxDailyHeatPlant_W_4D, dailyHeatDirect_Wh_4D
        
        
        #dimensions: [placements, SM, TES]
        annualHeat_Wh_3D = dailyHeatOutput_Wh_4D.sum(axis=0)
      
        self.opt_data['annualHeat_Wh_3D'] = annualHeat_Wh_3D #dimensions: [placements, SM, TES]
        self.opt_data['dailyHeatOutput_Wh_4D'] = dailyHeatOutput_Wh_4D #dimensions: [time(days), placements, SM, TES]

        # #stored heat per year
        # #dimensions: [placements, SM, TES]
        # annualHeatStored_Wh_3D = dailyHeatStored_Wh_4D.sum(axis=0)
        
        # #total heat storable per year
        # #dimensions: [placements, SM, TES]
        # annualHeatstoreable_Wh_3D = annualHeatStored_Wh_3D + directHeatUsage_Wh_3D
        
        # #max Heat input to PowerPlant
        # #dimensions: [placements, SM, TES]
        # maxAnnualHeatPlant_W_3D = np.tensordot(
        #     Powerplant_consumption_max_W_1D, #Powerplant_consumption_max_W_3D[0,:,:],
        #     np.ones(dimensions[3]),
        #     axes=0,
        # ) * aggregate_by_day.shape[1] # hours per  year
        
        # #total heat useable per year
        # #dimensions: [placements, SM, TES]
        # annualHeat_Wh_3D = np.minimum(annualHeatstoreable_Wh_3D, maxAnnualHeatPlant_W_3D)
        return self
    
    def calculateEconomics_Plant_Storage_4D(self):
        ''' Calculate the Capex for the Plant for placements, and variations of TES and SM

        Parameters
        ----------
        params : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        '''
        assert 'CAPEX_plant_cost_EUR_per_kW' in self.ptr_data.index
        assert 'CAPEX_storage_cost_EUR_per_kWh' in self.ptr_data.index
        assert 'CAPEX_indirect_cost_%_CAPEX' in self.ptr_data.index
        assert 'eta_powerplant_1' in self.ptr_data.index

        #Cost estiamtations for plant and storage:
        #dimensions: [SM, TES]
        sm_2D = np.tile(self.sm, (self.opt_data['dimensions'][3], 1)).T
        tes_2D = np.tile(self.tes, (self.opt_data['dimensions'][2], 1))
        
        #CAPEX_Plant_Storage per Solar field size
        #dimensions: [SM, TES]
        CAPEX_EUR_per_kW_SF_2D = (self.ptr_data['CAPEX_plant_cost_EUR_per_kW'] / sm_2D * self.ptr_data['eta_powerplant_1'] \
            + self.ptr_data['CAPEX_storage_cost_EUR_per_kWh'] * tes_2D / sm_2D) \
                * (1 + self.ptr_data['CAPEX_indirect_cost_%_CAPEX']/100)
        
        #yearly cost of storage and plant
        #dimensions: [SM, TES]
        CAPEX_EUR_per_a_kW_SF_2D = CAPEX_EUR_per_kW_SF_2D * self.sim_data['annuity']
        varOPEX_EUR_per_a_kW_SF_2D = CAPEX_EUR_per_kW_SF_2D * self.ptr_data['OPEX_%_CAPEX'] / 100
        fixOPEX_EUR_per_a_kW_SF_2D = 0
        del CAPEX_EUR_per_kW_SF_2D

        #dimensions: [SM, TES]
        TOTEX_Plant_storage_EUR_per_a_kw_SF_2D = CAPEX_EUR_per_a_kW_SF_2D + varOPEX_EUR_per_a_kW_SF_2D + fixOPEX_EUR_per_a_kW_SF_2D
        del CAPEX_EUR_per_a_kW_SF_2D, varOPEX_EUR_per_a_kW_SF_2D, fixOPEX_EUR_per_a_kW_SF_2D
        
        #TOTEX plant and storage per year (abolute)
        #Q_sf_des / 1000 * speccosts_EUR_per_kw_sf_2D
        #dimensions: [placements, SM, TES]
        TOTEX_Plant_storage_EUR_per_a_3D = np.tensordot(
            self.placements['capacity_sf_W_th'] / 1000,
            TOTEX_Plant_storage_EUR_per_a_kw_SF_2D,
            axes=0
            )
        
        #costs of field:
        #dimensions: [placements, SM, TES]
        TOTEX_SF_EUR_per_a_3D = np.tensordot(
            np.tensordot(
                self.placements['Totex_SF_EUR_per_a'].values,
                np.ones(self.opt_data['dimensions'][2]),
                axes=0,
            ),
            np.ones(self.opt_data['dimensions'][3]),
            axes=0,
        )
        #dimensions: [placements, SM, TES]
        TOTEX_CSP_EUR_per_a_3D = TOTEX_SF_EUR_per_a_3D + TOTEX_Plant_storage_EUR_per_a_3D
            

        # cost per heat
        #dimensions: [placements, SM, TES]
        # LCO_Heat_EUR_per_Wh = TOTEX_CSP_EUR_per_a_3D / self.opt_data['annualHeat_Wh_3D']
 
        self.opt_data['TOTEX_CSP_EUR_per_a_3D'] = TOTEX_CSP_EUR_per_a_3D #TODO: remove
        self.opt_data['TOTEX_SF_EUR_per_a_3D'] = TOTEX_SF_EUR_per_a_3D #TODO: remove
        self.opt_data['TOTEX_Plant_storage_EUR_per_a_3D'] = TOTEX_Plant_storage_EUR_per_a_3D

        return self
             
    def optimal_Plant_Configuration_4D(self):
        #Calculate average LCOE
        
        
        #dimensions: [time(days), placements, SM, TES]
        Q_plant_des_Wh_per_day_4D = np.tensordot(
            np.tensordot(
                np.tensordot(
                    np.ones(self.aggregate_by_day.shape[0]),   #days
                    self.placements['capacity_sf_W_th'],             #placements 
                    axes=0                                  
                ),
                1 / self.sm,                                   #SM
                axes=0
            ),
            np.ones(self.opt_data['dimensions'][3]),                       #TES
            axes=0
        ) * 24 #h/dey
        
        #dimensions: [time(days), placements, SM, TES]
        rel_load_plant_4D = 0.5 + 0.5 * (self.opt_data['dailyHeatOutput_Wh_4D'] / Q_plant_des_Wh_per_day_4D)
        
        #dimensions: [time(days), placements, SM, TES]
        efficiency_daily_averaged_1_4D = self._get_plant_efficiency(rel_load_plant=rel_load_plant_4D, eta_nom=self.ptr_data['eta_powerplant_1'])
        del rel_load_plant_4D
        
        efficiency_daily_averaged_1_4D = self.ptr_data['eta_powerplant_1']
        #dimensions: [time(days), placements, SM, TES]
        self.opt_data['dailyPowerOutput_Wh_4D'] = efficiency_daily_averaged_1_4D * self.opt_data['dailyHeatOutput_Wh_4D']
        del efficiency_daily_averaged_1_4D
        self.opt_data['annualPowerOutput_Wh_3D'] = self.opt_data['dailyPowerOutput_Wh_4D'].sum(axis=0)
        
        LCOE_EUR_per_Wh = self.opt_data['TOTEX_CSP_EUR_per_a_3D'] / self.opt_data['annualPowerOutput_Wh_3D']
        
        #find minimum:
        #dimensions: [placements]
        sm_opt = []
        tes_opt = []
        sm_opt_i = []
        tes_opt_i = []
        #loop placemnts(I did not find a function wich gives the argmin along two axes (1,2))
        for i in range(0, self.opt_data['dimensions'][1]):
            temp = LCOE_EUR_per_Wh[i,:,:]
            #find minimum index
            sm_opt_index, tes_opt_index = np.unravel_index(np.argmin(temp, axis=None), temp.shape)
            #append
            sm_opt.append(self.sm[sm_opt_index])
            tes_opt.append(self.tes[tes_opt_index])
            sm_opt_i.append(sm_opt_index)
            tes_opt_i.append(tes_opt_index)
        

        # set minimum values to placement df
        self.opt_data['LCOE_EUR_per_Wh'] = LCOE_EUR_per_Wh
        self.placements['sm_opt'] = sm_opt
        self.placements['tes_opt'] = tes_opt
        self.placements['storage_capacity_kWh_th'] = self.placements['capacity_sf_W_th'] / sm_opt * tes_opt / 1000
        self.placements['power_plant_capacity_W_el'] = self.placements['capacity_sf_W_th'] / sm_opt * self.ptr_data['eta_powerplant_1']
        
        #select the corresponding sm-colum from 'directHeatUsage_Wh_3D_ts'
        # self.sim_data['directHeatUsage_Wh'] = np.take_along_axis(
        #     arr=self.opt_data['directHeatUsage_Wh_3D_ts'],
        #     indices=np.array(sm_opt_i)[None,:,None],
        #     axis=2
        # ) #dims directHeatUsage_Wh_3D_ts: [time(hours), placements, SM]
        
        # self.sim_data['directHeatUsage_Wh'] = np.take_along_axis(
        #     arr=self.opt_data['dailyPowerOutput_Wh_4D'],
        #     indices=np.array(sm_opt_i)[None,:,None, None],
        #     axis=2
        # ) #dimensions 'dailyPowerOutput_Wh_4D': [time(days), placements, SM, TES]
        return self

    def calculate_electrical_output(self, onlynightuse=True, debug_vars = False):
        '''from sm and tes opt, calculate the electrical output.
            idea: as much energy as possible will be stored to be flexible,
            the rest is forced to be depending on solar radiation
        '''
    
        assert 'storage_capacity_kWh_th' in self.placements.columns    
        assert 'power_plant_capacity_W_el' in self.placements.columns
        
        pass
        
        #aggregate_by_day
        dt = (self._time_index_[1] - self._time_index_[0]) / pd.Timedelta(hours=1)
        if hasattr(self, 'aggregate_by_day'):
            aggregate_by_day = self.aggregate_by_day
        else:
            aggregate_by_day = np.eye(365).repeat(24, axis=1)
        HeattoPlant_per_day_Wh = np.einsum('ij,jk', aggregate_by_day, self.sim_data['HeattoPlant_W']) * dt
        Parasitics_plant_per_day_Wh_el = np.einsum('ij,jk', aggregate_by_day, self.sim_data['Parasitics_W_el']) * dt #issue #13
        Heat_heating_sf_daily_Wh_th = np.einsum('ij,jk', aggregate_by_day, self.sim_data['P_heating_W']) #issue #13
        
        # max thermal input the power plant is capable of
        if onlynightuse:
            operationalhours_per_day = np.einsum('ij,jk', aggregate_by_day, (self.sim_data['solar_zenith_degree']>90))
        else:
            operationalhours_per_day = 24
        power_plant_max_heat_Wh = self.placements['power_plant_capacity_W_el'].values / self.ptr_data['eta_powerplant_1'] * operationalhours_per_day
        
        #calculate stored and directly used heat per day
        # heat transfered into the storage (preferred, as max dispatchability is good)
        #limit by storage size
        Heat_stored_per_day_Wh = np.minimum(HeattoPlant_per_day_Wh, self.placements['storage_capacity_kWh_th']*1000)
        #limit by plant size
        Heat_stored_per_day_Wh = np.minimum(Heat_stored_per_day_Wh, power_plant_max_heat_Wh/self.ptr_data['storage_efficiency_1'])
        # heat transfered directly to the plant (2nd option)
        if onlynightuse:
            Heat_directly_per_day_Wh = 0
        else:
            Heat_directly_per_day_Wh = np.minimum(
                HeattoPlant_per_day_Wh - Heat_stored_per_day_Wh, # maximum heat abvailable
                power_plant_max_heat_Wh - Heat_stored_per_day_Wh*self.ptr_data['storage_efficiency_1'] #maximum heat capable for the plant (cf_day=1)
            )
        # heat output from the storage
        Heat_from_storage_per_day_Wh = np.maximum(Heat_stored_per_day_Wh * self.ptr_data['storage_efficiency_1'] - Heat_heating_sf_daily_Wh_th,0) #13
        P_backup_heating_daily_Wh_el = np.maximum(Heat_heating_sf_daily_Wh_th - Heat_stored_per_day_Wh * self.ptr_data['storage_efficiency_1'],0) #13
        Parasitics_plant_per_day_Wh_el += P_backup_heating_daily_Wh_el
        
        # total heat useable
        Heat_total_per_day_Wh = Heat_from_storage_per_day_Wh+Heat_directly_per_day_Wh
        assert (Heat_total_per_day_Wh <= power_plant_max_heat_Wh*1.001).all()
        if debug_vars:
            self.placements['avrg_sf_efficiency_1'] = self.sim_data['HeattoPlant_W'].sum(axis=0) / (self.placements['aperture_area_m2'] * self.sim_data['direct_normal_irradiance'].sum(axis=0))
            self.placements['Heat_after_curtailment_1'] = Heat_total_per_day_Wh.sum(axis=0) / self.sim_data['HeattoPlant_W'].sum(axis=0)
        # calculate rel load and efficiency
        

        # rel load is defined as the ratio of daily output to maximal output.
        # As the Poweplant wont output the total power over the whole day, the formula is corrected by:
        # rel_load* = 0.5 * 0.5 + rel_load
        rel_load_plant= (Heat_total_per_day_Wh / power_plant_max_heat_Wh)
        
        #Gafurov2015: 
        # rel_efficiency [1] = 54.92 + 112.73 * rel - 104.63 * rel^2 + 37.05 * rel^3
        #dimensions: [time(days), placements, SM, TES]
        efficiency_daily_averaged_1 = self._get_plant_efficiency(rel_load_plant=rel_load_plant, eta_nom=self.ptr_data['eta_powerplant_1'])
        del rel_load_plant
        
        #calculate gross power output: Heat * daily efficiency
        #calculate net power output : power gross - parasitic * share (share distributes the parasitic evenly to each output power)
        
        #bound
        Power_gross_bound_per_day_Wh = Heat_directly_per_day_Wh * efficiency_daily_averaged_1
        #with np.seterr(divide='ignore', invalid='ignore'):
        share = np.nan_to_num(Heat_directly_per_day_Wh/Heat_total_per_day_Wh, 0)
        Power_net_bound_per_day_Wh = Power_gross_bound_per_day_Wh - (Parasitics_plant_per_day_Wh_el * share)
        
        #dispatchable
        Power_gross_dispatchable_per_day_Wh = Heat_from_storage_per_day_Wh * efficiency_daily_averaged_1
        #with np.seterr(divide='ignore', invalid='ignore'):
        share = np.nan_to_num(Heat_from_storage_per_day_Wh/Heat_total_per_day_Wh, 0)
        Power_net_dispatchable_per_day_Wh = Power_gross_dispatchable_per_day_Wh - (Parasitics_plant_per_day_Wh_el * share)
        
        
        #add up for total output
        Power_net_total_per_day_Wh = Power_net_bound_per_day_Wh + Power_net_dispatchable_per_day_Wh
        
        if debug_vars:
            self.placements['mean_gross_turbine_efficiency_1'] = (Power_gross_dispatchable_per_day_Wh.sum(axis=0) + Power_gross_bound_per_day_Wh.sum(axis=0)) / Heat_total_per_day_Wh.sum(axis=0)
            self.placements['turbine_gross_to_net'] = Power_net_total_per_day_Wh.sum(axis=0) / (Power_gross_dispatchable_per_day_Wh.sum(axis=0) + Power_gross_bound_per_day_Wh.sum(axis=0))
            
        #get avrg cf
        Power_net_total_Wh_per_a = Power_net_total_per_day_Wh.sum(axis=0)
        steps_per_year = pd.Timedelta(hours=8760) / (self._time_index_[1] - self._time_index_[0])
        Power_cf = Power_net_total_Wh_per_a / (self.placements['power_plant_capacity_W_el'].values * steps_per_year)
        
        #save data
        self.sim_data_daily['Power_net_total_per_day_Wh'] = Power_net_total_per_day_Wh
        self.sim_data_daily['Power_net_bound_per_day_Wh'] = Power_net_bound_per_day_Wh
        self.placements['Power_net_total_Wh_per_a'] = Power_net_total_Wh_per_a
        self.placements['Power_net_bound_%_per_a'] = np.nan_to_num(Power_net_bound_per_day_Wh.sum(axis=0) / Power_net_total_Wh_per_a) * 100 #%
    
    def calculate_LCOE(self):
        '''calculates the LCOE from plant and storage sizes, SF totex and Net power output
        '''
        #calculate_economics
        TOTEX_EUR_per_a = self._get_totex_from_self()
        
        CAPEX_total_EUR = self._get_capex(
                A_aperture_m2=self.placements['aperture_area_m2'],
                A_land_m2=self.placements['land_area_m2'],
                Qdot_field_des_W=self.placements['capacity_sf_W_th'],
                eta_des_power_plant=self.ptr_data['eta_powerplant_1'],
                sm=self.placements['sm_opt'],
                tes=self.placements['tes_opt'],
                c_field_per_aperture_area_EUR_per_m2=self.ptr_data['CAPEX_solar_field_EUR_per_m^2_aperture'],
                c_land_per_land_area_EUR_per_m2=self.ptr_data['CAPEX_land_EUR_per_m^2_land'],
                c_storage_EUR_per_kWh_th=self.ptr_data['CAPEX_storage_cost_EUR_per_kWh'],
                c_plant_EUR_per_kW_el=self.ptr_data['CAPEX_plant_cost_EUR_per_kW'],
                c_indirect_cost_perc_per_direct_Capex=self.ptr_data['CAPEX_indirect_cost_%_CAPEX'],
            )
        #     #annual cost
        # CAPEX_total_EUR_per_a = CAPEX_total_EUR * self.sim_data['annuity']
        
        # OPEX_EUR_per_a = self._get_opex(
        #     CAPEX_total_EUR=CAPEX_total_EUR,
        #     OPEX_fix_perc_CAPEX_per_a=self.ptr_data['OPEX_%_CAPEX'],
        #     auxilary_power_Wh_per_a=self.sim_data['Parasitics_solarfield_W_el'].sum(axis=0),
        #     electricity_price_EUR_per_kWh=self.ptr_data['electricity_price_EUR_per_kWh'],
        # )
        
        # TOTEX_EUR_per_a = self._get_totex(
        #     CAPEX_total_EUR_per_a=CAPEX_total_EUR_per_a,
        #     OPEX_EUR_per_a=OPEX_EUR_per_a,
        # )
        
        self.placements['CAPEX_total_EUR'] = CAPEX_total_EUR
        self.placements['lcoe_EURct_per_kWh_el'] = TOTEX_EUR_per_a / self.placements['Power_net_total_Wh_per_a'] *1E2 * 1E3 #EUR/WH to EURct/kWh
    
    
    ### Try to increase speed of PV-Lib by dropping one loop. Aparently, multiple locations are not supported by PV-Lib. If there are performance
    ### issues, try again. So keep this in mind here 
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
    
    def _applyVariation(self):
        '''if DNI_factor and _T_offset in placements, manipulate the input time series in order to calculate the DNI/T_amb variation
        '''
        dni_factor_name = 'DNI_factor'
        t_amb_offset_name = 'T_offset_K'
        
        if dni_factor_name in self.placements.columns and t_amb_offset_name in self.placements.columns:
            # data want to be manipulated.
            # print warnings
            print('\n________________________\nWaring, Dev function started The results are not valid any more!\nOnly use this, if you are familiar with code.\nStarting to manipulate DNI and T_amb.')
            print('Factors for DNI [1]:')
            print(self.placements[dni_factor_name].unique())
            print('Offset for T_amb [K]:')
            print(self.placements[t_amb_offset_name].unique())
            
            #do manipulation
            f_DNI_np = np.tile(self.placements[dni_factor_name].values.T, (8760,1))
            self.sim_data['direct_normal_irradiance'] = self.sim_data['direct_normal_irradiance'] * f_DNI_np

            d_T_np = np.tile(self.placements[t_amb_offset_name].values.T, (8760,1))
            self.sim_data['surface_air_temperature'] = self.sim_data['surface_air_temperature'] + d_T_np

            #get annual values
            self.placements['mean_DNI_W_per_m2'] = self.sim_data['direct_normal_irradiance'].mean(axis=0)
            self.placements['mean_T_amb_K'] = self.sim_data['surface_air_temperature'].mean(axis=0)
        else:
            pass #no manipulation, should be default case
        
    def _get_capex(
        self,
        A_aperture_m2,
        A_land_m2,
        Qdot_field_des_W,
        eta_des_power_plant,
        sm,
        tes,
        c_field_per_aperture_area_EUR_per_m2,
        c_land_per_land_area_EUR_per_m2,
        c_storage_EUR_per_kWh_th,
        c_plant_EUR_per_kW_el,
        c_indirect_cost_perc_per_direct_Capex,
    ):
        '''calculate the initial capex in EUR (Invest)

        Parameters
        ----------
        A_aperture_m2 : [type]
            [description]
        A_land_m2 : [type]
            [description]
        Qdot_field_des_W : [type]
            [description]
        eta_des_power_plant : [type]
            [description]
        sm : [type]
            [description]
        tes : [type]
            [description]
        c_field_per_aperture_area_EUR_per_m2 : [type]
            [description]
        c_land_per_land_area_EUR_per_m2 : [type]
            [description]
        c_storage_EUR_per_kWh_th : [type]
            [description]
        c_plant_EUR_per_kW_el : [type]
            [description]
        c_indirect_cost_perc_per_direct_Capex : [type]
            [description]
        annuity : [type]
            [description]

        Returns
        -------
        np.nparray
            [description]
        '''
        
        Power_powerplant_des_W_el = Qdot_field_des_W / sm * eta_des_power_plant
        Heat_storage_des_kWh_th = Qdot_field_des_W / sm * tes
        CAPEX_Plant_EUR = c_plant_EUR_per_kW_el * Power_powerplant_des_W_el / 1000 #W --> kW
        
        #capex storage
        CAPEX_Storage_EUR = c_storage_EUR_per_kWh_th * Heat_storage_des_kWh_th / 1000 #Wh-->kWh
        
        #solar field
        CAPEX_Field_EUR = c_field_per_aperture_area_EUR_per_m2 * A_aperture_m2
        CAPEX_Land_EUR =  c_land_per_land_area_EUR_per_m2 * A_land_m2
        
        #indirect costs
        CAPEX_Indirect_EUR = (CAPEX_Plant_EUR + CAPEX_Storage_EUR + CAPEX_Field_EUR + CAPEX_Land_EUR) * c_indirect_cost_perc_per_direct_Capex/100
        
        CAPEX_total_EUR = (CAPEX_Plant_EUR + CAPEX_Storage_EUR + CAPEX_Field_EUR + CAPEX_Land_EUR + CAPEX_Indirect_EUR)
        
        return CAPEX_total_EUR
        
    def _get_opex(
        self,
        CAPEX_total_EUR,
        OPEX_fix_perc_CAPEX_per_a,
        auxilary_power_Wh_per_a,
        electricity_price_EUR_per_kWh    
    ):    
        '''calculate the opex

        Parameters
        ----------
        CAPEX_total_EUR : [type]
            [description]
        OPEX_fix_perc_CAPEX_per_a : [type]
            [description]
        auxilary_power_Wh_per_a : [type]
            [description]
        electricity_price_EUR_per_kWh : [type]
            [description]
        '''
        OPEX_fix_EUR_per_a = CAPEX_total_EUR * OPEX_fix_perc_CAPEX_per_a / 100
        OPEX_var_EUR_per_a = electricity_price_EUR_per_kWh * auxilary_power_Wh_per_a / 1000 * 1 # W --> kWh
        OPEX_EUR_per_a = OPEX_fix_EUR_per_a + OPEX_var_EUR_per_a
        return OPEX_EUR_per_a
    
    def _get_totex(
        self,
        CAPEX_total_EUR_per_a,
        OPEX_EUR_per_a,
    ):
        '''calculate totex from capex and opex

        Parameters
        ----------
        CAPEX_total_EUR_per_a : [type]
            [description]
        OPEX_EUR_per_a : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        '''
        TOTEX_EUR_per_a = CAPEX_total_EUR_per_a + OPEX_EUR_per_a
        return TOTEX_EUR_per_a
    
    def _get_totex_from_self(self, sm_manipulation = None, tes_manipulation = None):
        '''calcualtes CSP Totes per a

        Returns
        -------
        TOTEX_EUR_per_a: pd.DataFrame
        '''
        
        assert 'capacity_sf_W_th' in self.placements.columns
        assert 'aperture_area_m2' in self.placements.columns
        assert 'land_area_m2' in self.placements.columns
        
        assert hasattr(self, 'ptr_data')
        
        assert 'annuity' in self.sim_data.keys()
        assert 'Parasitics_solarfield_W_el' in self.sim_data.keys()
        
        
        #allow sm and tes manipulation
        if sm_manipulation == None:
            assert 'sm_opt' in self.placements.columns
            sm = self.placements['sm_opt']
        else: 
            assert isinstance(sm_manipulation, int) or isinstance(sm_manipulation, float)
            sm = sm_manipulation
        
        if tes_manipulation == None:
            assert 'tes_opt' in self.placements.columns
            tes = self.placements['tes_opt']
        else: 
            assert isinstance(tes_manipulation, np.int32) or isinstance(tes_manipulation, float) or isinstance(tes_manipulation, int)
            tes = tes_manipulation
        
        CAPEX_total_EUR = self._get_capex(
            A_aperture_m2=self.placements['aperture_area_m2'],
            A_land_m2=self.placements['land_area_m2'],
            Qdot_field_des_W=self.placements['capacity_sf_W_th'],
            eta_des_power_plant=self.ptr_data['eta_powerplant_1'],
            sm=sm,
            tes=tes,
            c_field_per_aperture_area_EUR_per_m2=self.ptr_data['CAPEX_solar_field_EUR_per_m^2_aperture'],
            c_land_per_land_area_EUR_per_m2=self.ptr_data['CAPEX_land_EUR_per_m^2_land'],
            c_storage_EUR_per_kWh_th=self.ptr_data['CAPEX_storage_cost_EUR_per_kWh'],
            c_plant_EUR_per_kW_el=self.ptr_data['CAPEX_plant_cost_EUR_per_kW'],
            c_indirect_cost_perc_per_direct_Capex=self.ptr_data['CAPEX_indirect_cost_%_CAPEX'],
        )
        #annual cost
        CAPEX_total_EUR_per_a = CAPEX_total_EUR * self.sim_data['annuity']
        
        OPEX_EUR_per_a = self._get_opex(
            CAPEX_total_EUR=CAPEX_total_EUR,
            OPEX_fix_perc_CAPEX_per_a=self.ptr_data['OPEX_%_CAPEX'],
            auxilary_power_Wh_per_a= 0,#self.sim_data['P_heating_W'].sum(axis=0),#self.sim_data['Parasitics_solarfield_W_el'].sum(axis=0), #not used, substracted from power plant output #issue #13
            electricity_price_EUR_per_kWh=self.ptr_data['electricity_price_EUR_per_kWh'],
        )
        
        TOTEX_EUR_per_a = self._get_totex(
            CAPEX_total_EUR_per_a=CAPEX_total_EUR_per_a,
            OPEX_EUR_per_a=OPEX_EUR_per_a,
        )
        return TOTEX_EUR_per_a
    
    
    def _get_plant_efficiency(
        self,
        rel_load_plant,
        eta_nom
    ):
        '''calculate the efficiency for the powerplant depending on its size

        Parameters
        ----------
        rel_load_plant : [type]
            [description]
        eta_nom : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        '''
        #Gafurov2015: 
        # rel_efficiency [1] = 54.92 + 112.73 * rel - 104.63 * rel^2 + 37.05 * rel^3
        eta_plant = (0.5492 \
                    + 1.1273 * rel_load_plant \
                    - 1.0463 * rel_load_plant**2 \
                    + 0.3705 * rel_load_plant**3) * eta_nom
        return eta_plant
    
    