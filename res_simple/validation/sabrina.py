import sys

from res_simple.weather import merra, noaa

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from os.path import basename
from scipy.stats.stats import pearsonr
from collections import namedtuple

class ValidationError(Exception): pass

def fetchDailyAverages(merraSource, path):
    # define lat/lon for our path
    station = noaa.gsodPathProfile(path)
    lat = station.LAT
    lon = station.LON
    year = station.YEAR
    
    # use our wind model
    windspeed = merra.hubWindSpeed( merraSource, lat=lat, lon=lon, height=10)

    # reading the station data from gsod
    gsod = noaa.parseGSOD(path, columns=['wind_speed'])
    
    # daily average -> must be changed for diffrent time steps like one year/ leapyear (Schaltjahr), 3 months, 100 days etc.
    startDay = pd.Timestamp(year=year, day=1, month=1)
    endDay = pd.Timestamp(year=year, day=31, month=12)
    days = pd.date_range(startDay, endDay, freq='D')
    
    dailyaverage = []
    for day in days:

        monthS = windspeed.index.month == day.month
        dayS = windspeed.index.day == day.day

        dailyaverage.append( windspeed[monthS & dayS].mean() )
        
    # done!
    output = pd.DataFrame({'gsod':gsod, 'model':dailyaverage}, index=days)
    
    bad = output.gsod.isnull() & output.gsod == 999.9
    
    return output[~bad]
    

Errors = namedtuple('Errors', "count abs rabs cum rcum corr")
def gsodError(res):
    
    sel = (~res.gsod.isnull()) & (res.gsod!=0)
    if sel.sum() == 0:
        return Errors(count=0, abs=0.0, rabs=0.0, cum=0.0, rcum=0.0, corr=0.0)
    
    # extract data
    gsod = res.gsod[sel]

    index = gsod.index
    model = res.model.loc[index]

    if model.isnull().any(): raise ValidationError("The modeled data is not complete")

    # total time series results
    cummulativeErrorTS = (gsod - model)
    cummulativeError=cummulativeErrorTS.mean()
    relCummulativeError = (100*cummulativeErrorTS/gsod).mean()
    
    absErrorTS = np.abs(gsod - model)
    absError=absErrorTS.mean()
    relAbsError = (100*absErrorTS/gsod).mean()
    
    if sel.sum()>1:
        corr = pearsonr(model, gsod)
    else:
        corr = (-9999,-9999)
    
    return Errors(count=int(sel.sum()), abs=absError, rabs=relAbsError, cum=cummulativeError, rcum=relCummulativeError, corr=corr[0])


ErrorSet = namedtuple('ErrorSet', 'file lat lon year winter spring summer fall')
def gsodErrorSet(path, merraSource):
    # Get modeled and measured dataz
    station = noaa.gsodPathProfile(path)
    windData = fetchDailyAverages(merraSource, path)
    
    # Define time contexts
    winter = (windData.index.month==1) | (windData.index.month==2) | (windData.index.month==12)
    spring = (windData.index.month==3) | (windData.index.month==4) | (windData.index.month==5)
    summer = (windData.index.month==6) | (windData.index.month==7) | (windData.index.month==8)
    fall = (windData.index.month==9) | (windData.index.month==10) | (windData.index.month==11)
    
    # Create a result dictionary
    results = {}
    
    # Evalueate errors
    results['file'] = basename(path)
    results['lat'] = station.LAT
    results['lon'] = station.LON
    results['year'] = gsodError(windData)
    results['winter'] = gsodError(windData[winter])
    results['spring'] = gsodError(windData[spring])
    results['summer'] = gsodError(windData[summer])
    results['fall'] = gsodError(windData[fall])
    
    # Done!
    return ErrorSet(**results)
