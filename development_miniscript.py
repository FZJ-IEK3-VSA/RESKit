#%%
import geokit as gk
import reskit as rk
import numpy as np
import matplotlib.pyplot as plt

#%%
#source_high_resolution=r'/storage/internal/data/gears/geography/wind/global_wind_atlas/GWA_3.0/gwa3_250_wind-speed_100m.tif'
source_low_resolution_gwa=rk.weather.GWAmeanSource.GWA_with_ERA5_pixel
era5_path_gwa = rk.weather.Era5Source.LONG_RUN_AVERAGE_WINDSPEED

source_low_resolution_dni=rk.weather.GSAmeanSource.DNI_with_ERA5_pixel
era5_path_dni = rk.weather.Era5Source.LONG_RUN_AVERAGE_DNI

#%%
matrix_GWA_lowres = gk.raster.extractMatrix(source_low_resolution_dni)
matrix_era5 =  gk.raster.extractMatrix(era5_path_dni)
matrix_GWA_lowres_cut = matrix_GWA_lowres[2:, 1:-1]
# %%
rel = matrix_era5 / matrix_GWA_lowres_cut /(1000 / 24)
# %%
np.histogram(np.nan_to_num(rel, nan=1))
plt.show()
#%%
plt.subplots(figsize=(16,10), dpi=1000)
plt.imshow(rel, cmap='RdBu', vmin=0, vmax=2)
plt.colorbar()
# %%


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=1.0, stop=2.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


from matplotlib import cm

RdBu = cm.get_cmap('RdBu', 50)
RdBu_sht = shiftedColorMap(cmap=RdBu, start=0, midpoint=1.0, stop=2.0, name='shiftedcmap')
# %%
