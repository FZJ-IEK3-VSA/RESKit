import pandas as pd
import geokit as gk
import re
import numpy as np
from collections import OrderedDict
from glob import glob

stations = None

def loadIsdHistory(path):
	globals()["stations"] = pd.read_excel(path) # load station data
	globals()["stations"].set_index(["USAF","WBAN"], inplace=True, drop=False) # set index

def wmoStation(USAF, WBAN=99999):
	# Make sure the isd-history dataset has been created
	if stations is None:
		raise RuntimeError("'stations' data has not been loaded properly. Make sure to call noaa.loadIsdHistory(..) with a path pointing to a copy of the latest 'isd-history.xlsx' file")

	# Return the location we need
	if not isinstance(USAF, int): USAF = int(USAF)
	if not isinstance(WBAN, int): WBAN = int(WBAN)
	return stations.loc[USAF,WBAN]

#############################################################
# ISD PARSING

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
	return raw[userColumns]


#############################################################
# GSOD PARSING

# make isd fields definitions
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
	return raw[userColumns]