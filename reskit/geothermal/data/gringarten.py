#%%
import numpy as np
import pandas as pd
import os

class   gringarten():
    
    #water properties
    rho_water = 1000 #kg/m^3
    cp_water = 4186 #J/kgK
    
    #rock properties
    rho_rock = 2550 #kg/m^3
    cp_rock = 1000 #J/kgK
    K_rock = 2.6 #W/mK

    SECONDS_PER_YEAR = 365*24*3600
    
    def __init__(self, Vdot_total, x, y, z, x_ED):
        self.Vdot_total = Vdot_total
        self.x = x
        self.y = y
        self.z = z
        
        self.x_ED = x_ED
        self.getNFracs()
    
    def getNFracs(self):
        if self.x_ED == "inf":
          n_Fracs = 1
        else:
          a = self.x_ED * self.K_rock * self.y * self.z / (self.rho_water * self.cp_water * self.Vdot_total) #definition of the dimensionles fracture space frim Augustine eg3
          
          n_Fracs = np.sqrt(self.x/(2*a)) # by geometry as a equals x_E/n and 2x_E*n=x        
        
        self.n_Fracs = n_Fracs
        
        #x_E = self.x / (2*n_Fracs)
        #x_ED = (self.rho_water * self.cp_water) / self.K_rock * self.Vdot_total / (n_Fracs * self.y * self.z) * x_E
        
    
    def getDimlessTime(self, time):
        self.time = time
        t_D = (self.rho_water * self.cp_water)**2 / (self.K_rock * self.rho_rock * self.cp_rock) * (self.Vdot_total / (self.n_Fracs * self.y * self.z))**2 * time
        self.t_D = t_D
    
    
    def getGringartenCurve(self, path=None):
        if path is None:
            path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "Gringartencurve.xlsx")
        
        dict_df = pd.read_excel(path, sheet_name=None)
        if self.x_ED == 2:
          df = dict_df["xED=2.0"]
        elif self.x_ED == 4:
          df = dict_df["xED=4.0"]
        elif self.x_ED == 8:
          df = dict_df["xED=8.0"]
        elif self.x_ED == "inf":
          df = dict_df["xED=inf"]
        
        td_grincurve = df['dimensionless time']
        Td_grincurve = df['dimensionless temperature']
        
        T_D = np.interp(
            x=self.t_D,
            xp=td_grincurve,
            fp=Td_grincurve,
            left=0,
            right=1,
        )
        
        T_D[T_D > 1] = 1
        T_D[T_D < 0] = 0
        
        self.T_D = T_D        
    
    def getWaterTemp(self, T_Rock, T_Inj):
        
        self.T_Rock = T_Rock
        self.T_Inj = T_Inj
        
        dT_Rock_Inj = T_Rock - T_Inj
        dT_Rock_Inj[dT_Rock_Inj<0] = np.nan

        T_water_outlet = np.expand_dims(T_Rock,2) - np.einsum('k,ij', self.T_D, dT_Rock_Inj)
        
        self.T_out = T_water_outlet
    
    def getEGSProps(self, timestep=None):

        dt = self.time[1] - self.time[0]
        Qdot_water = self.Vdot_total * self.rho_water * self.cp_water * (self.T_out - self.T_Inj)
        Q_water = Qdot_water.cumsum(axis=2) * dt
        mdot_water = self.Vdot_total * self.rho_water
        
        #Heat in place
        T_amb = 15 #°C
        Q_Rock_total = self.x * self.y * self.z * self.rho_rock * self.cp_rock * (self.T_Rock - T_amb) #* np.ones(Q_water.shape[2])
        Q_Rock_useable = self.x * self.y * self.z * self.rho_rock * self.cp_rock * (self.T_Rock - self.T_Inj)# * np.ones(len(Q_water))
        
        Q_Rock_unused =  np.expand_dims(Q_Rock_total,2) - Q_water
        T_Rock_avrg = Q_Rock_unused / (self.x * self.y * self.z * self.rho_rock * self.cp_rock) + T_amb

        dT_Rock_avrg = np.expand_dims(self.T_Rock,2) - T_Rock_avrg
        
        R_total = Q_water / np.expand_dims(Q_Rock_total, 2)
        
        if timestep is None:
            
            output = {
                'R_total': R_total,
                'Qdot_water': Qdot_water,
                'Q_water': Q_water,
                'Q_Rock_total': Q_Rock_total,
                'Q_Rock_useable': Q_Rock_useable,
                'Q_Rock_unused': Q_Rock_unused,
                'T_Rock_avrg': T_Rock_avrg,
                'T_Water_out': self.T_out,
                'dt_Rock_avrg': dT_Rock_avrg,     
                'mdot_water': mdot_water,
                'T_D': self.T_D,           
            }
        
        else:
            output = {
                'R_total': R_total[:,:,timestep-1],
                'Qdot_water': Qdot_water[:,:,timestep-1],
                'Q_water': Q_water[:,:,timestep-1],
                'Q_Rock_total': Q_Rock_total,
                'Q_Rock_useable': Q_Rock_useable,
                'Q_Rock_unused': Q_Rock_unused[:,:,timestep-1],
                'T_Rock_avrg': T_Rock_avrg[:,:,timestep-1],
                'T_Water_out': self.T_out[:,:,timestep-1],
                'dT_Rock_avrg': dT_Rock_avrg[:,:,timestep-1],                   
                'mdot_water': mdot_water,
                'T_D': self.T_D[timestep-1],
                
            }
        
        return output
    
    def getResourceUseTime(self, T_abandon):
        '''returns the time in years, after which the reservoir is depleted (if enough time steps are given)

        Parameters
        ----------
        T_abandon : int, float
            temperature at which the reservoir needs to be abandoned (eg. 150 degC)

        Returns
        -------
        float
            time in years until the abandon temperature is reached
        '''
        

        #for 1D: np.where((self.T_out > T_abandon)[9,0])[0].max()
        #for 1d: np.absolute(self.T_out - T_abandon)[9,0].argmin()
        #select the point in time where the water outlet temperature is closest to min useable temerature (min of abs(T_out - T_abandon))
        timestep_abandon = np.absolute(self.T_out - T_abandon).argmin(axis=2)
        #get the time for the time steps in years
        time_abandon = self.time[timestep_abandon] / self.SECONDS_PER_YEAR

        return time_abandon

#%%
if __name__ == "__main__":
    
    SECONDS_PER_YEAR = 365*24*3600
    #unit tests
    grin = gringarten(50E-3, 1000, 1000, 1000, 2)
    assert np.isclose(grin.n_Fracs, 8.9/2, rtol = 0.01) #from augstine with different mass flow
    grin.getDimlessTime(np.array([1*365*24*3600, 5*365*24*3600, 10*365*24*3600, 20*365*24*3600, 30*365*24*3600]))
    assert np.allclose(grin.t_D, np.array([0.0103537 , 0.05176851, 0.10353702, 0.20707403, 0.31061105]))
    
