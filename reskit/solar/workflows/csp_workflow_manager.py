from logging import error, warn, warning
from ...workflow_manager import WorkflowManager
import numpy as np
import pandas as pd
import pvlib
from numba import jit
import time
import geokit as gk
from typing import Union

class PTRWorkflowManager(WorkflowManager):
    def __init__(self, placements):
        """

        __init_(self, placements)

        Initialization of an instance of the generic SolarWorkflowManager class.

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

        self.sim_data['Heat_kW'] = np.multiply(self.placements['Area'].to_numpy(), self.sim_data['direct_horizontal_irradiance']) * area_usage * geometric_efficiency

        return self
    

    def adjust_variable_to_long_run_average(
        self,
        variable: str,
        source_long_run_average: Union[str, float, np.ndarray],
        real_long_run_average: Union[str, float, np.ndarray],
        real_lra_scaling: float = 1,
        spatial_interpolation: str = "linear-spline"):
        
        """Adjusts the average mean of the specified variable to a known long-run-average

        Note:
        -----
        uses the equation: variable[t] = variable[t] * real_long_run_average / source_long_run_average

        Parameters
        ----------
        variable : str
            The variable to be adjusted

        source_long_run_average : Union[str, float, np.ndarray]
            The variable's native long run average (the average in the weather file)
            - If a string is given, it is expected to be a path to a raster file which can be 
              used to look up the average values from using the coordinates in `.placements`
            - If a numpy ndarray (or derivative) is given, the shape must be one of (time, placements)
              or at least (placements) 

        real_long_run_average : Union[str, float, np.ndarray]
            The variables 'true' long run average
            - If a string is given, it is expected to be a path to a raster file which can be 
              used to look up the average values from using the coordinates in `.placements`
            - If a numpy ndarray (or derivative) is given, the shape must be one of (time, placements)
              or at least (placements)

        real_lra_scaling : float, optional
            An optional scaling factor to apply to the values derived from `real_long_run_average`. 
            - This is primarily useful when `real_long_run_average` is a path to a raster file
            - By default 1

        spatial_interpolation : str, optional
            When either `source_long_run_average` or `real_long_run_average` are a path to a raster 
            file, this input specifies which interpolation algorithm should be used
            - Options are: "near", "linear-spline", "cubic-spline", "average"
            - By default "linear-spline"
            - See for more info: geokit.raster.interpolateValues

        Returns
        -------
        WorkflowManager
            Returns the invoking WorkflowManager (for chaining)
        """

        if isinstance(real_long_run_average, str):
            real_lra = gk.raster.interpolateValues(
                real_long_run_average,
                self.locs,
                mode=spatial_interpolation)
            assert not np.isnan(real_lra).any() and (real_lra > 0).all()
        else:
            real_lra = real_long_run_average

        if isinstance(source_long_run_average, str):
            source_lra = gk.raster.interpolateValues(
                source_long_run_average,
                self.locs,
                mode=spatial_interpolation)
            assert not np.isnan(source_lra).any() and (source_lra > 0).all()
        else:
            source_lra = source_long_run_average

        self.sim_data[variable] *= real_lra * real_lra_scaling / source_lra
        print('Factors_from_LRA:')
        print(real_lra * real_lra_scaling / source_lra)
        print('__')
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
                    altitude=row.elev
                )


            self.sim_data['solar_zenith_degree'][:, location_iter] = _solarpos['zenith'].values

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


        pass

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


    def calculateHeattoHTF(self, eta_ptr_max: float = 0.742, eta_cleaness: float = 1, A_aperture_sf = 909060):
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
        # direct_horizontal_irradiance: W/m^2

        self.eta_ptr_max = eta_ptr_max
        self.eta_cleaness = eta_cleaness
        self.A_aperture_sf = A_aperture_sf

        self.sim_data['HeattoHTF_W'] = self.eta_ptr_max \
                                        * self.eta_cleaness \
                                        * np.cos(np.deg2rad(self.sim_data['theta'])) \
                                        * self.sim_data['IAM'] \
                                        * self.sim_data['eta_shdw'] \
                                        * self.sim_data['eta_wind'] \
                                        * self.sim_data['eta_degradation'] \
                                        * A_aperture_sf \
                                        * self.sim_data['direct_horizontal_irradiance']

        self.sim_data['P_DNI'] = A_aperture_sf * self.sim_data['direct_horizontal_irradiance']

        self.sim_data['P_DNI_eta_opt'] = A_aperture_sf * self.sim_data['direct_horizontal_irradiance'] * eta_ptr_max


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
            assert 'A' in params.keys()
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
            assert 'direct_horizontal_irradiance' in self.sim_data.keys()


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
                    losses: np.ndarray, K: np.ndarray, DNI: np.ndarray, A: float, b:np.ndarray, relTMplant: float,
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
                    losses: np.ndarray, K: np.ndarray, DNI: np.ndarray, A: float, b:np.ndarray, relTMplant: float,
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
                        DNI=self.sim_data['direct_horizontal_irradiance'],
                        A=params['A'],
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
                        DNI=self.sim_data['direct_horizontal_irradiance'],
                        A=params['A'],
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
                PL_plant_fix: Fixed plant losses in % of design point power output of te plant
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

            Q_sf_des = nominal_sf_efficiency * self.A_aperture_sf * params['I_DNI_nom'] * (1-nominal_receiver_heat_losses) #W
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


    def simpleStorageAndPlant():
        pass #TODO: 6h TES, 2,5SM, efficency from Gafurov and Trieb to calculate stirage size, plant size and power output (no spaceial resolution


    def calclateEconomics(self, WACC: float = 0.08, lifetime: float = 25, A_Aperture = -1, SF_density_total = 0.38, calculationmethod: str = 'franzmann2021', params: dict = {}):
        
        #Check inputs
        land_size = A_Aperture / SF_density_total
        
        # Calculate annuity factor from WACC and lifetime like in Heuser
        annuity = (WACC * (1 + WACC)**lifetime) / ((1+WACC)**lifetime - 1)
        
        # calculate the average annual heat production
        #self.sim_data['annualHeat'] = self.sim_data['HeattoPlant_W'].mean(axis=0)  * (pd.Timedelta(days=365) / (self._time_index_[-1] - self._time_index_[0]))
        
        if calculationmethod == 'franzmann2021':
            assert 'CAPEX_solar_field_USD_per_m^2_aperture' in params.keys, "'CAPEX_solar_field_USD_per_m^2_aperture' needs to be in params"
            assert 'CAPEX_land_USD_per_m^2_land' in params.keys, "'CAPEX_land_USD_per_m^2_land' needs to be in params"
            assert 'fixOPEX_%_CAPEX_per_a' in params.keys, "'_CAPEX_per_a' needs to be in params"
            assert 'indirect_cost_%_CAPEX' in params.keys, "'indirect_cost_USD' needs to be in params"
            
            CAPEX_sf_USD = self.A_aperture_sf * params['CAPEX_solar_field_USD_per_m^2_aperture'] \
                        + self.land_size / params['CAPEX_land_USD_per_m^2_land'] \
                        + params['CAPEX_land_USD_per_m^2_land']
            
            
            
        elif False:
            pass

        self.sim_data['capexSF'] = investSF * annuity
        self.sim_data['opexFix'] = 0
        self.sim_data['opexVar'] = 0

        self.sim_data['totex'] = self.sim_data['capexSF'] + self.sim_data['opexFix'] +self.sim_data['opexVar']

        self.sim_data['LCO_heat'] = self.sim_data['totex'] / self.sim_data['annualHeat']

        

        return self

        


    ### Try to increase speed of PV-Lib by dropping one loop. Aparently, multiple locations are not supported by PV-Lib. If there are performance
    ### issues, try again. So keep this in mind here 
    #
    # def calculateSolarPositionfaster(self):
    #     """
    #     DOES NOT WORK PV LIP DOES NOT SUPPORT MULTIPLE LOCATIONS
    #     calculates the solar position in terms of hour angle and declination from time series and location series of the current object
    #
    #     Returns:
    #         [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for sim_data['hour_angle'][timeserie_iter, location_iter] and 
    #         sim_data['declination_angle'][timeserie_iter, location_iter]]
    #     """
    #     # Shape Input data to 1D
    #     _time = pd.DatetimeIndex(np.tile(self.time_index.values, self._numlocations))
    #     _dayoftheyear = pd.DataFrame(np.tile(self.time_index.day_of_year.values, self._numlocations))
    #     _latitute = pd.DataFrame(self.placements['lat'].values.repeat(self._numtimesteps))
    #     _longitude = pd.DataFrame(self.placements['lon'].values.repeat(self._numtimesteps))
    #
    #
    #
    #     #calculate the solar hour angle with pv lib
    #     _hour_angle = pvlib.solarposition.hour_angle(
    #                 times=_time,
    #                 longitude=_longitude,
    #                 equation_of_time=pvlib.solarposition.equation_of_time_pvcdrom(dayofyear=_dayoftheyear)
    #                 )
    #                 # equation_of_time is the deviation betwee real time and solar time
    #
    #     self.sim_data['hour_angle_degree_fast'] = np.reshape(
    #         a=_hour_angle,
    #         newshape=(self._numtimesteps, self._numlocations)
    #     )
    #  
    #
    #
    #     #calculate the solar declonation angle with pv lib
    #     _declination_angle_degree = \
    #          pvlib.solarposition.declination_cooper69(dayofyear=_dayoftheyear) * 180 / np.pi
    #
    #
    #     self.sim_data['declination_angle_degree_fast'] = np.reshape(
    #         a=_declination_angle_degree,
    #         newshape=(self._numtimesteps, self._numlocations)
    #     )



    #     #calculate the solar altitide angle
    #     _solarpos = pvlib.solarposition.get_solarposition(
    #         time=_time,
    #         latitude=_latitute,
    #         longitude=_longitude
    #     )

    #     _solar_zenith_degree = _solarpos['zenith'].values

    #     truetracking_angles = pvlib.tracking.singleaxis(
    #         apparent_zenith=_solarpos['apparent_zenith'],
    #         apparent_azimuth=_solarpos['azimuth'],
    #         axis_tilt=0,
    #         axis_azimuth=180,
    #         max_angle=90,
    #         backtrack=False,  # for true-tracking
    #         gcr=0.5)  # irrelevant for true-tracking


    #     #from [1]	KALOGIROU, Soteris A. Environmental Characteristics. In: Soteris Kalogirou, ed. Solar energy engineering. Processes and systems. Waltham, Mass: Academic Press, 2014, pp. 51-123.
    #     # fromula 2.12
    #     _solar_altitude_angle = np.rad2deg(np.arcsin(np.cos(_solar_zenith_degree)))

    #     self.sim_data['solar_altitude_angle_degree_fast'] = np.reshape(
    #         a=_declination_angle_degree,
    #         newshape=(self._numtimesteps, self._numlocations)
    #     )

    #     self.sim_data['aoi'] = np.reshape(
    #         a=truetracking_angles['aoi'],
    #         newshape=(self._numtimesteps, self._numlocations)
    #     )

    #     return self