import pandas as pd
import geokit as gk
import re
import numpy as np
from collections import OrderedDict, namedtuple
from glob import glob
from os.path import basename

isdStations = None
igraStations = None

def loadIsdStations(path):
    globals()["isdStations"] = pd.read_csv(path) # load station data
    globals()["isdStations"].set_index(["USAF","WBAN"], inplace=True, drop=False) # set index

def loadIGRAStations(path):
    colSpecs = OrderedDict()

    colSpecs["ID"] = (0,11)
    colSpecs["lat"] = (12,20)
    colSpecs["lon"] = (21,30)
    colSpecs["elev"] = (31,37)
    colSpecs["name"] = (38,71)
    colSpecs["startYear"] = (72,76)
    colSpecs["endYear"] = (77,81)
    colSpecs["count"] = (82,88)

    # load station data
    globals()["igraStations"] = pd.read_fwf(path, colSpecs=[v for k,v in colSpecs.items()], names=[k for k in colSpecs.keys()])
    globals()["igraStations"].set_index("ID", inplace=True, drop=False) # set index

def wmoStation(USAF, WBAN=99999):
    # Make sure the isd-history dataset has been created
    if isdStations is None:
        raise RuntimeError("'isdStations' data has not been loaded properly. Make sure to call noaa.loadIsdHistory(..) with a path pointing to a copy of the latest 'isd-history.xlsx' file")

    # Return the location we need
    if not isinstance(USAF, int): USAF = int(USAF)
    if not isinstance(WBAN, int): WBAN = int(WBAN)
    return isdStations.loc[USAF,WBAN]

def igraStation(ID):
    # Make sure the isd-history dataset has been created
    if isdStations is None:
        raise RuntimeError("'igraStations' data has not been loaded properly. Make sure to call noaa.loadIGRAHistory(..) with a path pointing to a copy of the latest 'isd-history.xlsx' file")

    # Return the location we need
    return igraStations.loc[USAF,WBAN]

#########################################
# Handel GSOD files
gsodProfiler = re.compile("(?P<USAF>[0-9]+)-(?P<WBAN>[0-9]+)-(?P<YEAR>[0-9]+).op")
def gsodPathProfile(path):
    # Make sure the isd-history dataset has been created
    if isdStations is None:
        raise RuntimeError("'isdStations' data has not been loaded properly. Make sure to call noaa.loadIsdHistory(..) with a path pointing to a copy of the latest 'isd-history.xlsx' file")

    base = basename(path)
    p = gsodProfiler.match(base)
    
    if p is None: raise RuntimeError('Path is bad :(')
        
    USAF = int(p.groupdict()['USAF'])
    WBAN = int(p.groupdict()['WBAN'])
    
    output = isdStations.loc[USAF, WBAN].copy()
    output['YEAR'] = int(p.groupdict()['YEAR'])

    return output

# make gsod fields definitions
gsodFields = OrderedDict()
gsodFields["usaf"]=(0,6), None, lambda x: x.astype(int)
gsodFields["wban"]=(7,12), None, lambda x: x.astype(int)
gsodFields["year"]=(14,18), None, lambda x: x.astype(int)
gsodFields["month"]=(18,20), None, lambda x: x.astype(int)
gsodFields["day"]=(20,22), None, lambda x: x.astype(int)

gsodFields["air_temp"]=(24,30), "9999.9", lambda x: (x.astype(float) - 32) / 1.8 # F -> C
gsodFields["air_temp_count"]=(31,33), None, lambda x: x.astype(int)

gsodFields["dew_temp"]=(35,41), "9999.9", lambda x: (x.astype(float) - 32) / 1.8 # F -> C
gsodFields["dew_temp_count"]=(42,44), None, lambda x: x.astype(int)

gsodFields["pressure"]=(57,63), "9999.9", lambda x: x.astype(float) # mbar
gsodFields["pressure_count"]=(64,66), None, lambda x: x.astype(int)

gsodFields["visibility"]=(68,73), "999.9", lambda x: x.astype(float) * 1609.34 # miles -> meters
gsodFields["visibility_count"]=(74,76), None, lambda x: x.astype(int)

gsodFields["wind_speed"]=(78,83), "999.9", lambda x: x.astype(float) * 0.514444 # knots -> m/s
gsodFields["wind_speed_count"]=(84,86), None, lambda x: x.astype(int)
gsodFields["wind_speed_max"]=(88,93), "999.9", lambda x: x.astype(float) * 0.514444 # knots -> m/s
gsodFields["wind_speed_gust"]=(95,100), "999.9", lambda x: x.astype(float) * 0.514444 # knots -> m/s

gsodFields["air_temp_max"]=(102,108), "9999.9", lambda x: (x.astype(float) - 32) / 1.8 # F -> C
gsodFields["air_temp_max_flag"]=(108,109), None, None
gsodFields["air_temp_min"]=(110,116), "9999.9", lambda x: (x.astype(float) - 32) / 1.8 # F -> C
gsodFields["air_temp_min_flag"]=(116,117), None, None

gsodFields["precipitation"]=(118,123), "99.99", lambda x: x.astype(float) * 2.54 # inches -> cm
gsodFields["precipication_flag"]=(123,124), None, None

gsodFields["snow_depth"]=(125,130), "999.9", lambda x: x.astype(float) * 2.54 # inches -> cm
gsodFields["FRSHTT"]=(132,138), None, None

def parseGSOD(path, columns='measurements'):
    """Parses mandatory data from Integrated Surface Data (ISD) files"""

    # Make sure columns list is good
    if columns == 'all': userColumns = list(gsodFields.keys())
    elif columns == 'measurements': userColumns = ["air_temp","dew_temp","pressure","visibility","wind_speed","wind_speed_max","wind_speed_gust","air_temp_max","air_temp_min","precipitation","snow_depth"]
    else: columns = userColumns = list(columns)

    totalColumns = list(set(["year","month","day"]).union(userColumns))

    # make raw data
    raw = pd.read_fwf(path, [gsodFields[k][0] for k in totalColumns], header=0, names=totalColumns,
                      converters=dict([(k,str) for k in totalColumns]), # make sure everything is read in as strings 
                                                                        #  (makes reading no-data easier)
                      na_values=dict([(k,v[1]) for k,v in gsodFields.items()]))

    # Fix Columns
    for c in totalColumns:
        if not gsodFields[c][2] is None:
            try:
                raw[c] = gsodFields[c][2](raw[c])
            except Exception as e:
                print(c)
                raise e

    # create datetime series
    raw.index = [pd.Timestamp(year=r.year, month=r.month, day=r.day) for r in raw.itertuples()]

    # done!
    if len(userColumns)==1: return raw[userColumns[0]]
    else: return raw[userColumns]

#############################################################
# Handel ISD files
# make isd fields definitions
isdFields = OrderedDict()
isdFields["chars"]=(0,4), None, lambda x: x.astype(int)
isdFields["usaf"]=(4,10), None, None
isdFields["wban"]=(10,15), None, None
isdFields["year"]=(15,19), None, lambda x: x.astype(int)
isdFields["month"]=(19,21), None, lambda x: x.astype(int)
isdFields["day"]=(21,23), None, lambda x: x.astype(int)
isdFields["hour"]=(23,25), None, lambda x: x.astype(int)
isdFields["minute"]=(25,27), None, lambda x: x.astype(int)
isdFields["source"]=(27,28), "9", None
isdFields["lat"]=(28,34), "99999", lambda x: x.astype(float)/1000
isdFields["lon"]=(34,41), "999999", lambda x: x.astype(float)/1000
isdFields["report"]=(41,46), "99999", None
isdFields["elev"]=(46,51), "9999", lambda x: x.astype(float)
isdFields["call_number"]=(51,56), "99999", None
isdFields["quality_process"]=(56,60), None, None

isdFields["wind_dir"]=(60,63), "999", lambda x: x.astype(float)
isdFields["wind_dir_quality"]=(63,64), None, None
isdFields["wind_dir_type"]=(64,65), "9", None
isdFields["wind_speed"]=(65,69), "9999", lambda x: x.astype(float)/10
isdFields["wind_speed_quality"]=(69,70), None, None

isdFields["ceiling"]=(70,75), "99999", lambda x: x.astype(float)
isdFields["ceiling_quality"]=(75,76), None, lambda x: x.astype(int)
isdFields["ceiling_determination"]=(76,77), "9", None
isdFields["ceiling_and_visability_ok"]=(77,78), "9", None

isdFields["visability"]=(78,84), "999999", lambda x: x.astype(float)
isdFields["visability_quality"]=(84,85), None, None
isdFields["visability_variable"]=(85,86), "9", None
isdFields["visability_variable_quality"]=(86,87), None, lambda x: x.astype(int)

isdFields["air_temp"]=(87,92), "9999", lambda x: x.astype(float)/10
isdFields["air_temp_quality"]=(92,93), None, None
isdFields["dew_temp"]=(93,98), "9999", lambda x: x.astype(float)/10
isdFields["dew_temp_quality"]=(98,99), None, None

isdFields["pressure"]=(99,104), "99999", lambda x: x.astype(float)/10
isdFields["pressure_quality"]=(104,105), None, None

def parseISD(path, columns='measurements'):
    """Parses mandatory data from Integrated Surface Data (ISD) files"""

    # Make sure columns list is good
    if columns == 'all': userColumns = list(isdFields.keys())
    elif columns == 'measurements': userColumns = ["wind_dir","wind_speed","ceiling","visability","air_temp","dew_temp","pressure"]
    else: columns = userColumns = list(columns)

    totalColumns = list(set(["year","month","day","hour","minute"]).union(userColumns))

    # make raw data
    raw = pd.read_fwf(path, [isdFields[k][0] for k in totalColumns], header=None, names=totalColumns,
                      converters=dict([(k,str) for k in totalColumns]), # make sure everything is read in as strings 
                                                                        #  (makes reading no-data easier)
                      na_values=dict([(k,v[1]) for k,v in isdFields.items()]))

    # Fix Columns
    for c in totalColumns:
        if not isdFields[c][2] is None:
            try:
                raw[c] = isdFields[c][2](raw[c])
            except Exception as e:
                print(c)
                raise e

    # create datetime series
    raw.index = [pd.Timestamp(year=r.year, month=r.month, day=r.day, hour=r.hour, minute=r.minute) for r in raw.itertuples()]

    # done!
    if len(userColumns)==1: return raw[userColumns[0]]
    else: return raw[userColumns]



#############################################################
# Handel IGRA files
# make isd fields definitions
IGRAHeaderLine = namedtuple("HeaderLine", "HEADREC ID YEAR MONTH DAY HOUR RELTIME_HOUR RELTIME_MIN NUMLEV P_SRC NP_SRC LAT LON")
def splitIGRAHeaderLine(line):
    HEADREC = line[0]
    ID = line[1:12]
    YEAR = int(line[13:17])
    MONTH = int(line[18:20])
    DAY = int(line[21:23])
    HOUR = int(line[24:26])
    RELTIME_HOUR = int(line[27:29]) 
    RELTIME_MIN = int(line[29:31])
    NUMLEV = int(line[32:36])
    P_SRC = line[37:45]
    NP_SRC = line[46:54]
    LAT = int(line[55:62])/10000.0
    LON = int(line[63:71])/10000.0

    return IGRAHeaderLine( HEADREC, ID, YEAR, MONTH, DAY, HOUR, RELTIME_HOUR, RELTIME_MIN, NUMLEV, P_SRC, NP_SRC, LAT, LON )

IGRADataLine = namedtuple("DataLine", "LVLTYP1 LVLTYP2 ETIME_MIN ETIME_SEC PRESS PFLAG GPH ZFLAG TEMP TFLAG RH DPDP WDIR WSPD")
def splitIGRADataLine(line):
    LVLTYP1 = int(line[0])
    LVLTYP2 = int(line[1])

    tmp = line[3:6]
    ETIME_MIN = int(tmp) if tmp != "   " else 0
    ETIME_SEC = int(line[6:8])
    PRESS = int(line[9:15]) # Pa
    PFLAG = line[15]
    GPH = int(line[16:21]) # m
    ZFLAG = line[21]
    TEMP = int(line[22:27])/10.0 # deg. C
    TFLAG = line[27]
    RH = int(line[28:33])/10.0 # percent
    DPDP = int(line[34:39])/10.0 # deg. C
    WDIR = int(line[40:45]) # degrees
    WSPD = int(line[46:51])/10.0 # m/s

    return IGRADataLine(LVLTYP1,LVLTYP2,ETIME_MIN,ETIME_SEC,PRESS,PFLAG,GPH,ZFLAG,TEMP,TFLAG,RH,DPDP,WDIR,WSPD)

Sounding = namedtuple("Sounding","header data")
def parseIGRA(path, minYear=1900, maxYear=100000):
    """Parses mandatory data from Integrated Surface Data (ISD) files"""

    # make empty data container
    data = OrderedDict()
    data["time"] = []
    data["lat"] = []
    data["lon"] = []
    data["ID"] = []
    data["pres"] = []
    data["gph"] = []
    data["temp"] = []
    data["rh"] = []
    data["wspd"] = []
    data["wdir"] = []
    data["pres_flag"] = []
    data["gph_flag"] = []
    data["temp_flag"] = []
    data["etime_flag"] = []

    # split into sounding groups
    f = open(path)

    while True:
        line = f.readline()
        if line == "": break
        if not line[0] == "#": raise RuntimeError("Expected a header line")
        h = splitIGRAHeaderLine(line)

        # Check for bad values
        if h.RELTIME_HOUR == 99: releaseHour = h.HOUR
        else: releaseHour = h.RELTIME_HOUR
        
        if h.RELTIME_MIN == 99: releaseMin = 0
        else: releaseMin = h.RELTIME_MIN

        release = pd.Timestamp(year=h.YEAR, month=h.MONTH, day=h.DAY, hour=releaseHour, minute=releaseMin)
        
        for i in range(h.NUMLEV):
            dataLine = f.readline()
            if h.YEAR < minYear or h.YEAR>maxYear: continue
            # parse the data line
            try:
                d = splitIGRADataLine(dataLine)
            except ValueError as e:
                print(dataLine)
                raise e 

            # check for bad values
            eFlag = False
            if d.ETIME_MIN == -88 or d.ETIME_MIN == -99: 
                elapsedMin = 0
                eFlag=True
            else: 
                elapsedMin = d.ETIME_MIN

            if d.ETIME_SEC == 88 or d.ETIME_SEC == 99: 
                elapsedSec = 0 
                eFlag=True
            else:
                elapsedSec = d.ETIME_SEC

            if d.PRESS == -9999: pressure = np.nan
            else: pressure = d.PRESS

            if d.GPH == -8888 or d.GPH == -9999: continue # dont bother with data we dont have a height for

            if d.TEMP < 0: temperature = np.nan
            else: temperature = d.TEMP

            if d.RH < 0: relHumidity = np.nan
            else: relHumidity = d.RH

            if d.WDIR == -8888 or d.WDIR == -9999: wdir = np.nan
            else: wdir = d.WDIR

            if d.WSPD < 0: wspd = np.nan
            else: wspd = d.WSPD

            # append to container
            try:
                data["time"].append(release + (pd.Timedelta(minutes=elapsedMin)+pd.Timedelta(seconds=elapsedSec)))
            except Exception as e:
                print(dataLine)
                print(h)
                print(d)
                raise e

            data["lat"].append(h.LAT)
            data["lon"].append(h.LON)
            data["ID"].append(h.ID)
            data["pres"].append(pressure)
            data["gph"].append(d.GPH)
            data["temp"].append(temperature)
            data["rh"].append(relHumidity)
            data["wspd"].append(wspd)
            data["wdir"].append(wdir)
            data["pres_flag"].append(d.PFLAG)
            data["gph_flag"].append(d.ZFLAG)
            data["temp_flag"].append(d.TFLAG)
            data["etime_flag"].append(eFlag)
    f.close()
    
    # output as dataframe
    output = pd.DataFrame(data)
    #output.set_index("time", inplace=True, drop=True)
    
    return output