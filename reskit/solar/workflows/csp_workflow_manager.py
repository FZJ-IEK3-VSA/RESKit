from logging import warn, error, warning
from ...workflow_manager import WorkflowManager
import numpy as np
import pandas as pd
import pvlib
from numba import jit
import time

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
    

    def calculateSolarPosition(self):
        """calculates the solar position in terms of hour angle and declination from time series and location series of the current object

        Returns:
            [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for sim_data['hour_angle'][timeserie_iter, location_iter] and 
            sim_data['declination_angle'][timeserie_iter, location_iter]]
        """


        #set up empty array
        self.sim_data['hour_angle_degree'] = np.empty(shape=(self._numtimesteps, self._numlocations))
        self.sim_data['declination_angle_degree'] = np.empty(shape=(self._numtimesteps, self._numlocations))
        self.sim_data['solar_zenith'] = np.empty(shape=(self._numtimesteps, self._numlocations))






        # iterate trough all location
        for location_iter, row in enumerate(self.placements[['lon', 'lat']].itertuples()):
            


            #calculate the solar hour angle with pv lib
            self.sim_data['hour_angle_degree'][:, location_iter] = \
                pvlib.solarposition.hour_angle(
                    times=self.time_index,
                    longitude=row.lon,
                    equation_of_time=pvlib.solarposition.equation_of_time_pvcdrom(dayofyear=self.time_index.day_of_year.values)
                    )
                    # equation_of_time is the deviation betwee real time and solar time



            #calculate the solar declonation angle with pv lib
            self.sim_data['declination_angle_degree'][:, location_iter] = \
                pvlib.solarposition.declination_cooper69(dayofyear=self.time_index.day_of_year.values) * 180 / np.pi


            #calculate the solar altitide angle
            self.sim_data['solar_zenith'][:, location_iter] = \
                pvlib.solarposition.get_solarposition(
                    time=self.time_index,
                    latitude=row.lat,
                    longitude=row.lon
                )['zenith'].values


            #from [1]	KALOGIROU, Soteris A. Environmental Characteristics. In: Soteris Kalogirou, ed. Solar energy engineering. Processes and systems. Waltham, Mass: Academic Press, 2014, pp. 51-123.
            # fromula 2.12
            self.sim_data['solar_altitude_angle_degree'] = np.rad2deg(np.arcsin(np.cos(np.deg2rad(self.sim_data['solar_zenith']))))


        pass

        return self


    def calculateSolarPositionfaster(self):
        """
        DOES NOT WORK PV LIP DOES NOT SUPPORT MULTIPLE LOCATIONS
        calculates the solar position in terms of hour angle and declination from time series and location series of the current object

        Returns:
            [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for sim_data['hour_angle'][timeserie_iter, location_iter] and 
            sim_data['declination_angle'][timeserie_iter, location_iter]]
        """
        # Shape Input data to 1D
        _time = pd.DatetimeIndex(np.tile(self.time_index.values, self._numlocations))
        _dayoftheyear = pd.DataFrame(np.tile(self.time_index.day_of_year.values, self._numlocations))
        _latitute = pd.DataFrame(self.placements['lat'].values.repeat(self._numtimesteps))
        _longitude = pd.DataFrame(self.placements['lon'].values.repeat(self._numtimesteps))



        #calculate the solar hour angle with pv lib
        _hour_angle = pvlib.solarposition.hour_angle(
                    times=_time,
                    longitude=_longitude,
                    equation_of_time=pvlib.solarposition.equation_of_time_pvcdrom(dayofyear=_dayoftheyear)
                    )
                    # equation_of_time is the deviation betwee real time and solar time

        self.sim_data['hour_angle_degree_fast'] = np.reshape(
            a=_hour_angle,
            newshape=(self._numtimesteps, self._numlocations)
        )



        #calculate the solar declonation angle with pv lib
        _declination_angle_degree = \
             pvlib.solarposition.declination_cooper69(dayofyear=_dayoftheyear) * 180 / np.pi


        self.sim_data['declination_angle_degree_fast'] = np.reshape(
            a=_declination_angle_degree,
            newshape=(self._numtimesteps, self._numlocations)
        )



        #calculate the solar altitide angle
        _solar_zenith = pvlib.solarposition.get_solarposition(
            time=_time,
            latitude=_latitute,
            longitude=_longitude
        )['zenith'].values


        #from [1]	KALOGIROU, Soteris A. Environmental Characteristics. In: Soteris Kalogirou, ed. Solar energy engineering. Processes and systems. Waltham, Mass: Academic Press, 2014, pp. 51-123.
        # fromula 2.12
        _solar_altitude_angle = np.rad2deg(np.arcsin(np.cos(_solar_zenith)))

        self.sim_data['solar_altitude_angle_degree_fast'] = np.reshape(
            a=_declination_angle_degree,
            newshape=(self._numtimesteps, self._numlocations)
        )


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
        assert 'hour_angle_degree' in self.sim_data.keys()
        assert 'declination_angle_degree' in self.sim_data.keys()
        assert 'solar_altitude_angle_degree' in self.sim_data.keys()
        assert 'lat' in list(self.placements.columns)




        # write data all locations to one 1-D array
        _hour_angle = self.sim_data['hour_angle_degree'].flatten(order='F')
        _declination = self.sim_data['declination_angle_degree'].flatten(order='F')
        _solar_altitude = self.sim_data['solar_altitude_angle_degree'].flatten(order='F')
        _latitude = self.placements['lat'].repeat(self.time_index.shape[0])

        #degree to radians
        _hour_angle = np.deg2rad(_hour_angle)
        _declination = np.deg2rad(_declination) 
        _solar_altitude = np.deg2rad(_solar_altitude)
        _latitude = np.deg2rad(_latitude) 

        #calculate the cos of theta for a north / south oreintation
        #formula (6) from Song How to decide the alignment of the parabolic trough collector according to the local latitude 2013
        
        if orientation in ['northsouth', 'song2013']:

            Sz = np.cos(_latitude) * np.cos(_declination) * np.cos(_hour_angle) + np.sin(_latitude) * np.sin(_declination)
            Sy = np.cos(_latitude) * np.sin(_declination) - np.sin(_latitude) * np.cos(_declination) * np.cos(_hour_angle)

            costheta_northsouth = 1- ( Sz / np.sqrt(np.square(Sy) + np.square(Sz)) )

            # when sun altidude is negative (after sunset), set values to 0
            costheta_northsouth = costheta_northsouth * (_solar_altitude >0)


            # costheta_northsouth = 1 - ((np.cos(_latitude) * np.cos(_declination) * np.cos(_hour_angle) + np.sin(_latitude) * np.sin(_declination))\
            #     / np.sqrt( np.square(np.cos(_latitude) * np.sin(_declination) - np.sin(_latitude) * np.cos(_declination) * np.cos(_hour_angle))
            #                 + np.square(np.cos(_latitude) * np.cos(_declination) * np.cos(_hour_angle) +  np.sin(_latitude) * np.sin(_declination))))
            
            # when sun sets, the result get higher than 1. set those values to 0
            



        #calculate the cos of theta for a east/ west oreintation
        #formula (7) from Song How to decide the alignment of the parabolic trough collector according to the local latitude 2013
        
        if orientation in ['eastwest', 'song2013']:
            
            #calculation
            costheta_eastwest = np.sqrt( (np.square(np.sin(_declination))-1) * np.square(np.sin(_hour_angle)) + 1 )

            # when sun altidude is negative (after sunset), set values to 0
            costheta_eastwest = costheta_eastwest * (_solar_altitude >0)



        if orientation == 'northsouth':
            costheta = costheta_northsouth #forced northsouth orientation
        elif orientation == 'eastwest':
            costheta = costheta_eastwest #forced eastwest orientation
        elif orientation == 'song2013':
            # apply song 2013:
            # if latitude is between -46° and +46°, use northsouth orientation, else eastwest orientation
            _isns = np.logical_and(_latitude < 46.06 /180*np.pi, _latitude > -46.06 /180*np.pi)

            costheta = _isns * costheta_northsouth + np.logical_not(_isns) * costheta_eastwest

        self.costheta_northsouth = costheta_northsouth
        self.costheta_eastwest = costheta_eastwest
        self.costheta = costheta
        self._isns = _isns

        # change array to 2D
        self.sim_data['costheta'] = np.reshape(
            a=costheta.values,
            newshape=(self._numtimesteps, self._numlocations),
            order='F'
        )

        return 


    def calculateIAM(self, a1: float = 0.000884 , a2: float = 0.00005369, a3: float = 0):
        """ Calculates the IAM angle modifier from incidence angle. Formula and default values are from: 
        [1]	GAFUROV, Tokhir, Julio USAOLA, and Milan PRODANOVIC. Modelling of concentrating
        solar power plant for power system reliability studies [online]. IET Renewable Power
        Generation. 2015, 9(2), 120-130. Available from: 10.1049/iet-rpg.2013.0377. 

        Args:
            a1 (float, optional): [description]. Defaults to 0.000884.
            a2 (float, optional): [description]. Defaults to 0.00005369.
            a3 (float, optional): [description]. Defaults to 0.

        Returns:
            [CSPWorkflowManager]: [Updated CSPWorkflowManager with new value for sim_data['IAM'][timeserie_iter, location_iter]
        """

        # check for input data availability
        assert 'costheta' in list(self.sim_data.keys())


        _theta = np.rad2deg(np.arccos(self.sim_data['costheta'])) # deg

        #calculate with formula for IAM: IAM = 1 + 'sum over i'  a_i * (theta^i / cos (theta))
        
        # a_i, theta in deg
        _IAM = 1 + a1 * np.power(_theta, 1) / self.sim_data['costheta'] \
            + a2 * np.power(_theta, 2) / self.sim_data['costheta'] \
            + a3 * np.power(_theta, 3) / self.sim_data['costheta']

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
            self.sim_data['eta_shdw'] = np.sin(np.deg2rad(self.sim_data['solar_altitude_angle_degree'])) / ( SF_density * self.sim_data['costheta'] )

        return self


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


    def calculateHeattoHTF(self, eta_ptr_max: float = 0.742, eta_cleaness: float = 1,A_aperture_sf = 909060):
        """Calculates the heat from Collector to heat transfer fluid. The result is before the heat losses of the HTF.

        Args:
            eta_ptr_max (float, optional): [Value for optical efficency of trough mirror and absorber]. Defaults to 0.742.
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



        self.sim_data['HeattoHTF_W'] = eta_ptr_max \
                                        * eta_cleaness \
                                        * self.sim_data['costheta'] \
                                        * self.sim_data['IAM'] \
                                        * self.sim_data['eta_shdw'] \
                                        * self.sim_data['eta_wind'] \
                                        * A_aperture_sf \
                                        * self.sim_data['direct_horizontal_irradiance']

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

        elif calculationmethod == "dersch2018":

            #check for input
            assert 'b' in params.keys()
            assert 'A' in params.keys()
            assert 'TM_plant' in params.keys()
            assert 'maxHTFTemperature' in params.keys()
            assert 'JITaccelerate' in params.keys()
            assert 'minHTFTemperature' in params.keys()
            assert 'inletHTFTemperature' in params.keys()
            assert params['b'].shape == (5,)

            assert 'IAM' in self.sim_data.keys()
            assert 'costheta' in self.sim_data.keys()
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
            _K = self.sim_data['costheta'] * self.sim_data['IAM']
            # heating power needed:
            _P_heating = np.zeros_like(self.sim_data['HeattoHTF_W'])
            # heat to Plant
            _HeattoPlant = np.zeros_like(self.sim_data['HeattoHTF_W'])
    

            #define simulation function
            def simulation(HeattoHTF: np.ndarray, temperature: np.ndarray, ambient_temperature: np.ndarray,
                losses: np.ndarray, K: np.ndarray, DNI: np.ndarray, A: float, b:np.ndarray, TM_plant: float,
                deltat: float, maxHTFTemperature: float, minHTFTemperature: float, inletHTFTemperature: float,
                p_heating: np.ndarray):
                """[Transient simulation of the HTF fluid temperature. Calculate bosses by empiric formulation as in Greenius]

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


                for i in range(0, temperature.shape[0]-1):
                    #deltaT = (T_out - T in) / - T_amb  ...  (in, out from the solar field view, defined in Greenius)
                    deltaT = temperature[i, :] - ambient_temperature[i, :]
                    #loss formula from greenius
                    losses[i, :] = K[i, :] * b[0] * deltaT * A * DNI[i, :] + A * ( b[1] * deltaT**1 + b[2] * deltaT**2 + b[3] * deltaT**3 + b[4] * deltaT**4)

                    #calculate temperature from energy balance around all thermal masses which need to be heated up (see sam manual)
                    temperature[i+1, :] = temperature[i, :] + (HeattoHTF[i, :] - losses[i, :]) * deltat / TM_plant

                    #maximal temperature is achieved, when outlet temperature is at max temperature.
                    temperature[i+1, :] = np.minimum(temperature[i+1, :], (maxHTFTemperature + inletHTFTemperature)/2)
                    
                    #when temperature is below minimal temperature, there will be an electrical heating for the HTF so that:
                    # 1) temperatere is locked at min temp
                    temperature[i+1, :] = np.maximum(temperature[i+1, :], minHTFTemperature)
                    # 2) there are parasitic losses for heating
                    p_heating[i, :] = losses[i, :]



                return temperature, losses, p_heating


            #second try
            def simulation(HeattoHTF: np.ndarray, temperature: np.ndarray, ambient_temperature: np.ndarray,
                losses: np.ndarray, K: np.ndarray, DNI: np.ndarray, A: float, b:np.ndarray, TM_plant: float,
                deltat: float, maxHTFTemperature: float, minHTFTemperature: float, inletHTFTemperature: float,
                P_heating: np.ndarray, HeattoPlant: np.ndarray):
                """[Transient simulation of the HTF fluid temperature. Calculate bosses by empiric formulation as in Greenius]

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
                    # deltaT = (T_out - T in) / - T_amb  ...  (in, out from the solar field view, defined in Greenius)
                    deltaT = temperature[i, :] - ambient_temperature[i, :]
                    # loss formula from greenius
                    losses[i, :] = K[i, :] * b[0] * deltaT * A * DNI[i, :] + A * ( b[1] * deltaT**1 + b[2] * deltaT**2 + b[3] * deltaT**3 + b[4] * deltaT**4)

                    # calculate temperature from energy balance around all thermal masses which need to be heated up (see sam manual)
                    temperature[i+1, :] = temperature[i, :] + (HeattoHTF[i, :] - losses[i, :]) * deltat / TM_plant

                    # maximal temperature is achieved, when outlet temperature is at max temperature.
                    temperature[i+1, :] = np.minimum(temperature[i+1, :], maxHTFmeanTemperature)
                    
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
                        TM_plant=params['TM_plant'],
                        deltat=deltat,
                        maxHTFTemperature=params['maxHTFTemperature'],
                        minHTFTemperature=params['minHTFTemperature'],
                        inletHTFTemperature=params['inletHTFTemperature'],
                        P_heating = _P_heating,
                        HeattoPlant = _HeattoPlant
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
                        TM_plant=params['TM_plant'],
                        deltat=deltat,
                        maxHTFTemperature=params['maxHTFTemperature'],
                        minHTFTemperature=params['minHTFTemperature'],
                        inletHTFTemperature=params['inletHTFTemperature'],
                        P_heating = _P_heating,
                        HeattoPlant = _HeattoPlant
                    )

            toc = time.time()
            print('Simulation time: ' + str(toc-tic) + 's.')



            #store data
            self.sim_data['HTF_mean_temperature_C'] = _temperature
            self.sim_data['Heat_Losses'] = _losses
            self.sim_data['P_heating_W'] = _P_heating
            self.sim_data['HeattoPlant_W'] = _HeattoPlant

        elif calculationmethod == "exact":
            warn('Wrong calculation for heat losses of heat transfer fluid selected. Losses will be set to zero.')
            _losses = np.zeros_like(self.sim_data['HeattoHTF_W'], dtype = float)
            self.sim_data['HeattoPlant_W'] = self.sim_data['HeattoHTF_W'] - _losses

        else:
            warn('Wrong calculation for heat losses of heat transfer fluid selected. Losses will be set to zero.')
            _losses = np.zeros_like(self.sim_data['HeattoHTF_W'], dtype = float)
            self.sim_data['HeattoPlant_W'] = self.sim_data['HeattoHTF_W'] - _losses
        
        self.sim_data['HeatLosses_W'] = _losses

        return self