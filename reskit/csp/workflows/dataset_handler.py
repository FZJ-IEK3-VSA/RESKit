import geokit as gk
import pandas as pd
from pandas.core.frame import DataFrame
import reskit as rk
import time
import numpy as np
import os
from reskit.csp.data import csp_data_path

class dataset_handler():
    
    def __init__(self, datasets) -> None:
        assert isinstance(datasets, list)
        
        self.datasets = datasets
        self.split_placements = None
    
    def split_placements(self, placements, gsa_dni_path, gsa_tamb_path):
        
        assert 'lat' in placements.columns
        assert 'lon' in placements.columns
        
        placements['geom'] = placements[['lon', 'lat']].apply(lambda x: gk.geom.Point(x[0], x[1], srs=4326))
        
        placements['dni_gsa'] = gk.raster.interpolate_Values(
            source=gsa_dni_path,
            points=placements.geom
        ) * 1000/24       
        
        placements['tamb_gsa'] = gk.raster.interpolate_Values(
            source=gsa_tamb_path,
            points=placements.geom
        ) 
        
        mat_HTF_opt = self._get_opt_HTF_matrix()
        
        placements['HTF_opt'] = placements[['dni_gsa', 'tamb_gsa']].apply(
            lambda x: self._lookup(x[0], x[1], mat_HTF_opt)
        )
        placements = placements.drop(['geom', 'dni_gsa', 'tamb_gsa'])
        placements_Heliosol = placements[placements.HTF_opt == 'H']
        placements_SolarSalt = placements[placements.HTF_opt == 'S']
        placements_Therminol = placements[placements.HTF_opt == 'T']
        
        return placements_Heliosol, placements_SolarSalt, placements_Therminol
    
    def _get_path_htf_opt(self):
        
        datasets = self.datasets
        def _list_to_str(li):
            s = ''
            for l in li:
                l = l.replace('Dataset_', '').split('_')[0]
                if s == '':
                    s = l
                else:
                    s += '_' + l
            return s
        
        path = os.path.join(
            csp_data_path,
            f'optimal_htf_selection{_list_to_str(datasetnames)}.csv'
        )
        return path

    
    def _get_opt_HTF_matrix(self):

        path = self._get_path_htf_opt()
        if os.path.isfile(path):
            htf_opt_matrix = pd.read_csv(path, index_col=[0], header=[0])
            htf_opt_matrix.index = htf_opt_matrix.index.astype(int)
            htf_opt_matrix.columns = htf_opt_matrix.columns.astype(int)
        else:
            print('No opt HTF matrix found. Calculating new Matrix.')
            htf_opt_matrix = self._calc_opt_HTF_matrix()
            htf_opt_matrix.index = htf_opt_matrix.index.astype(int)
            htf_opt_matrix.columns = htf_opt_matrix.columns.astype(int)
        return htf_opt_matrix
    
    
    def _calc_opt_HTF_matrix(self) -> pd.DataFrame:
        '''calculates the optimal htf for a variation of t_amb and dni and stores it inside reskit

        Returns
        -------
        [pd.DataFrame]
            [matrix with opt htf for different T_amb and DNIs]
        '''
        dT_vector = np.arange(-40,30,10)
        fDNI_vector = [.1, .25, .5, .75, 1.0, 1.25, 1.5]
        dT_matrix = np.tile(dT_vector, (len(fDNI_vector),1))
        fDNI_matrix = np.tile(fDNI_vector, (len(dT_vector),1)).T

        n_placements = fDNI_matrix.size

        #noor 2 ptr plant, morocco
        placements = pd.DataFrame()
        placements['lon'] = [-6.8] * n_placements     # Longitude
        placements['lat'] = [31.0] * n_placements  # Latitude
        placements['area'] = [1E6] * n_placements
        placements['T_offset_K'] = dT_matrix.flatten()
        placements['DNI_factor'] = fDNI_matrix.flatten()

        print('placements:', len(placements))
        
    
        era5_path=r'C:\Users\d.franzmann\data\ERA5\7\6'
        era5_path=r'/storage/internal/data/gears/weather/ERA5/processed/4/7/6/2015'
        datasetnames = self.datasets
        global_solar_atlas_dni_path = 'default_local'

        for datasetname in datasetnames:
            print('datasetname', datasetname)
            
            out = rk.csp.workflows.workflows.CSP_PTR_ERA5_specific_dataset(
                placements = placements,
                era5_path = era5_path,
                global_solar_atlas_dni_path = global_solar_atlas_dni_path,
                datasetname = datasetname,
                return_self=True,
                JITaccelerate = False,
                verbose = True,
                debug_vars = False,
                onlynightuse=True,
                fullvariation=False,
                #output_variables=['lcoe_EURct_per_kWh_el']
            )
            out = out.placements[['mean_T_amb_K', 'mean_DNI_W_per_m2', 'lcoe_EURct_per_kWh_el']]
            
            if datasetname == 'Dataset_Heliosol_2030':
                data_heliosol = out
            elif datasetname == 'Dataset_SolarSalt_2030':
                data_solarsalt = out
            elif datasetname == 'Dataset_Therminol_2030':
                data_therminol = out
        
        variable_name = 'lcoe_EURct_per_kWh_el'
        pivot_Heliosol = data_heliosol.pivot(index='mean_T_amb_K', columns='mean_DNI_W_per_m2', values=variable_name)
        pivot_SolarSalt = data_solarsalt.pivot(index='mean_T_amb_K', columns='mean_DNI_W_per_m2', values=variable_name)
        pivot_Therminol = data_therminol.pivot(index='mean_T_amb_K', columns='mean_DNI_W_per_m2', values=variable_name)
        
        
        pivot_SolarSalt[pivot_SolarSalt<0] = 1E9
        pivot_SolarSalt[np.isinf(pivot_SolarSalt)] = 1E9
        
        min_Heliosol = (pivot_Heliosol < pivot_SolarSalt) & (pivot_Heliosol <pivot_Therminol)
        min_SolarSalt = (pivot_SolarSalt < pivot_Heliosol) & (pivot_SolarSalt <pivot_Therminol)
        min_Therminol = (pivot_Therminol < pivot_SolarSalt) & (pivot_Therminol <pivot_Heliosol)

        min = pivot_SolarSalt * min_SolarSalt \
            + pivot_Heliosol * min_Heliosol\
            + pivot_Therminol * min_Therminol
        
        def _getstringdf(str, like):
            out = like.copy()
            out[:]= str
            return out
        
        argmin =  _getstringdf('S', min_Heliosol) * min_SolarSalt \
            + _getstringdf('H', min_Heliosol) * min_Heliosol \
            +   _getstringdf('T', min_Heliosol) * min_Therminol
            
        argmin.to_csv(self._get_path_htf_opt())

        return argmin
    
    def _lookup(x,y,table):
        
        x_index = min(table.columns, key= lambda c: abs(c-x))
        y_index = min(table.index, key= lambda i: abs(i-y))
        
        return table.loc[y_index,x_index]
    

    
    

if __name__ == '__main__':
    
    datasetnames = ['Dataset_Heliosol_2030', 'Dataset_SolarSalt_2030', 'Dataset_Therminol_2030']
    
    d = dataset_handler(datasets=datasetnames)
    htf_opt_matrix = d._get_opt_HTF_matrix()

    placements = pd.DataFrame()
    n_placements = 2
    placements['tamb_gsa'] = [-6.8] * n_placements     # Longitude
    placements['dni_gsa'] = [310.0] * n_placements  # Latitude
    placements['area'] = [1E6] * n_placements
    
    
    
        
    pass