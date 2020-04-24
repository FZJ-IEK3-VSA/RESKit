import numpy as np
import geokit as gk
from os.path import isfile

from ..util import ResError


def location_to_tilt(locs, convention="Ryberg2020", **k):
    """Simple system tilt estimators based off latitude and longitude coordinates

    'convention' can be...
        * "Ryberg2020"
            - The following equation is used:
                42.327719357601396 * arctan[ 1.5 * abs(latitude) ]
        * A string consumabe by 'eval'
            - Can use the variable 'latitude'
            - Ex. "latitude*0.76"
        * A raster file

    """
    locs = gk.LocationSet(locs)

    if convention == 'Ryberg2020':
        tilt = 42.327719357601396 * np.arctan(1.5 * np.radians(np.abs(locs.lats)))

    elif isfile(convention):
        tilt = gk.raster.interpolateValues(convention, locs, **k)

    else:
        try:
            tilt = eval(convention, {}, {"latitude": locs.lats})
        except:
            raise ResError("Failed to apply tilt convention")

    return tilt
