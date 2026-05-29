# ATTRIBUTION: This module is adapted from HPSCTerrSys/ParFlow_DataRetrievalAndExtraction
# Source: https://github.com/HPSCTerrSys/ParFlow_DataRetrievalAndExtraction
# License: MIT
# Original functionality preserved for extracting discharge data from THREDDS-based ParFlow data.

"""
This module extracts 1D (timeseries) data from gridded netCDF files staged 
on a remote THREDDS server. Data can be extracted for multiple positions on
the grid (longitude, latitude, and depth) and multiple variables.

PURPOSE:
The tool is a fully applicable data retrieval and extraction software to get 
timeseries information for user-specified points of interest (POIs) from netCDF
data stored on a THREDDS server or locally.
It furthemore serves as a demonstrator and example for an interaction with the 
server and datasets.
The tool has many generic apsects and is not limited to ParFlow data extraction.

In general, the extraction from THREDDS may take a lilttle while, depending on
how busy the server is and the type of exraction and netCDF file properties. 

INPUTS:
- JSON run control file, see examples, with information on location and
  depth (if applicable) for which data is to be extracted, the dataset URL
  on the THREDDS data server (could also be a local file on the filesystem)
- Output format, either 'csv' or 'var', the variable is always returned, i.e.,
  setting 'var' does not change anything. Setting 'csv' in addition also exports
  one csv file per record. 
- The JSON file contains a ParFlow specific file with soil hydraulic property
  information and the specification of the model grid as longitude and latitude
  arrays. This file is needed and specific for a setup of the model system. 
  ('indicatorFile')
- netCDF, model output, aligned with CMOR standard (e.g. all dimensions need to 
  ne defined) file as specified in the JSON runctrl-file.
All inputs must be specified at all times.

OUTPUTS:
- Returns ndarray, float, records are along the y-axis, time (days) are along
  the x-axis. This array is always returned. If there is only a single record
  per JSON run-control file, still the 2D array is created. Each row in the 
  array represents one record per JSON file. 
- If requested, also returns a csv file, one csv file per record in the JSON run
  control file. The file might be imported into any spreadsheet tool. 
  Compatibility might be hampered by decimal seperators used by the spreadsheet 
  software. The filename of the csv file is automatically generated from the 
  specified data and retrieval options. 

FUNCTIONALITIES:
- This tool finds the closest grid elements of the model grid to a location or 
  point of interest as defined in a runcontrol file and extracts a timeseries
  at the respective grid element.
- There is an additional check whether a locaiton might be on a lake, river or 
  an ocean grid element.
- Even if there is no valid data at a grid element, the data for this grid 
  element is returned, albeit with a warning message.
- The JSON runctrl-files are a versatile means to set up individual extraction
  schedules, e.g., single variable, single location/depth, but various 
  initialisation times, or single varibale, many different locations, a number
  of depths, e.g., to compare to your own measurements, single variable, 
  location, depth but subsequent years per record creates a multi-annual 
  timeseries, etc.
- As time information needs to be changed in case of a regular usage, the user 
  is expected to know how to (efficiently) modify the JSON files for their 
  needs.
- By specifiying different depths in subsequent records in the JSON file, e.g., 
  timeseries might be extracted, which along the y-axis feature a vertical 
  profile for the POI.
- The extracted csv files contain a set of metadata to not confuse them with 
  each other.
- The nerest neighbour search algorithm as specified also allows to extract 
  timeseries fo rmultiple neighbouring locations.

USAGE:
data = data_extraction_tool.data_extraction(<ctrl_file>, <output_format>)

USAGE EXAMPLES:
data = data_extraction_tool.data_extraction('$HOME/example5_1Loc_1Var_forecast.json', 'csv')

LIMITATIONS:
- No check whether the specified URL pointing to netCDF file on THREDDS
  server is actually OK. If you use the example JSON files for the 
  DE06 forecasts or climatology, make sure the data URL is specified OK and 
  data actually exist on the THREDDS server.
- Assumes that the variable to be extracted is the last variable in the 
  structure of the netCDF file. Variable can also be manually set.
- Per JSON file multiple records are possible; each record can feature
  a specific location, site name, depth, variable (forecast or climatology,
  2D or 3D); the individually extracted timeseries (number of timeseries)
  equals the number of records in the JSON file; the individual timeseries
  are combine dinto a 2D numpy array (see OUTPUTS); therefore it is not
  possible to combine timeseries with a different length, e.g., from the
  climatology data (usually 365 days per year in the annual files) and the 
  daily forecasts (9 days into the future). Also, if you want to combine 
  several years for a single location, this is not possible if a leapyear
  is included.
"""

import numpy as np
from netCDF4 import Dataset
from netCDF4 import num2date

import sys
import datetime
import csv
import json

__author__ = "Suad HAMMOUDEH, Klaus GEORGEN"
__copyright__ = "Copyright 2024, http://www.fz-juelich.de"
__credits__ = [""]
__license__ = "MIT"
__version__ = "v1.0.0"
__maintainer__ = "Suad HAMMOUDEH"
__email__ = "s.hammoudeh@fz-juelich.de"
__status__ = "Production"

def spher_dist(lon1, lat1, lon2, lat2, Rearth=6371.0):

    """
    Calculate the spherical / haversine distance of the location of the point 
    of interest (POI) from all grid elements of the model grid to determine the 
    closest model grid element to the POI.

    INPUTS:
    - lon1: ndarray, float, longitude array in [rad] of model grid
    - lat1: ndarray, float, latitude array in [rad] of model grid
    - lon2: ndarray, float, longitude array in [rad] of POI
    - lat2: ndarray, float, latitude array in [rad] of POI
    - Rearth: float, Earth radius [km]

    OUTPUTS:
    - spherical_distance: ndarray, float, [km], array of distances of all
      model grid elements to the POI

    SOURCE: 
    - https://www.kompf.de/gps/distcalc.html
    """

    spherical_distance = Rearth * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))

    return spherical_distance

def find_depth_index(depth, soil_layer_bnds):

    """
    Given the depth information provided for the extraction, find the
    layer for which the depth is specified, or, depending on the 
    variable, find the lower boundary of a depth layer, the depth is
    specified for. No depth ranges are possible at this time.
    In case of a flux variable, the flux is always given for the 
    interface between two depth layers at or below the specified depth
    value. In case of state variables, an index is returtned which 
    provides value for the depth layer the specified depth is in; if the 
    depth is specified at lower boundary of this layer, the information
    for the layer above is provided. 
    
    INPUTS:
    - depth (float), soil depth [m], positive value
    - soil_layer_bnds (float), lower boundaries of the ParFlow soil 
      layers

    OUTPUTS:
    - depth_idx: integer
      index of the depth layer (either the layer or the lower boundary)
    """

    # this is the lower boundaries of the ParFlow depth layers
    # ParFlow vertical coordinate starts with index 0 at the lowermost
    # model level, i.e., level 0 is 60-42m depth, level 14 is 0.02-0.0m
    # 15 layers in total, given here: soil_layer_bnds
    # this information is part of the netCDF file
    depth_list = soil_layer_bnds.tolist()
    #depth_list = [60.0, 42.0, 27.0, 17.0, 7.0, 3.0, 2.0, 1.3, 0.8, 0.5, 0.3, 0.17, 0.1, 0.05, 0.02]
    depth_list.append(0.0)
    print(depth_list)
   
    if depth > max(depth_list):
        print("the specified depth value is out of the range of the dataset, min depth, max depth: ", min(depth_list), max(depth_list))
        print("exit now")
        sys.exit()

    # if the specified depth is within a depth layer, find the layer's 
    # upper and lower boundaries, if it matches a boundary, provide the
    # index for the boundary (in case of flux variables)
    for i in range(len(depth_list) - 1):
        #print(i)
        if depth_list[i] >= depth > depth_list[i + 1]:
            upper_boundary = depth_list[i + 1]
            lower_boundary = depth_list[i]
            print("your specified depth lies within a depth-layer/on a boundary: showing boundaries and returning index for the layer")
            print(lower_boundary, upper_boundary)
            break
    depth_idx = depth_list.index(lower_boundary)
    print(depth_idx, depth_list[depth_idx])

    return depth_idx, lower_boundary, upper_boundary

def data_extraction(runctrl_file, output_format):

    """
    Returns data as specified in JSON file, either as numpy variable or as a 
    CSV file.
    """

    # extracts information from the control JSON file, loop over records
    # of the JSON file, each record might feature a different variable,
    # i.e., THREDDS server filename, depth, and/or location
    with open(runctrl_file, 'r') as f:

        data = json.load(f)
        indicator = data['indicatorFile']
        locations = data['locations']
        
        # get the longitude and latiude arrays from the auxilliary dataset
        # which defines the model grid and some soil hydraulic properties
        # assumes you are only dealing with single experiment or model 
        # setup, just read once
        # indicator[time, depth, lon, lat]
        with Dataset(indicator, 'r') as ncIndicator:
            simLons = ncIndicator.variables['lon'][:]
            simLats = ncIndicator.variables['lat'][:]
            indicator = ncIndicator.variables['Indicator'][0, 14, :, :]

        # loop over the records in the JSON file
        # each record may contain a separate file URL on the THREDDS 
        # server
        i = 0
        for ilocation in locations:

            print("----------------------------------------") 
            print("JSON record:", i)

            locationID = ilocation["locationID"]
            locationLon = ilocation["locationLon"]
            locationLat = ilocation["locationLat"]
            locationDepth = ilocation["depth"]
            locationData = ilocation["simData"]
            print(f'location ID, lon, lat, depth, sim-data: {locationID}, {locationLon}, {locationLat}, {locationDepth}, {locationData}')

            # finding the array index of the longitude and latitude pair of 
            # interest, i.e., the POI, based on 2D sheperical distance array
            # dim0=y, dim1=x
            dist = spher_dist(np.deg2rad(simLons), np.deg2rad(simLats), np.deg2rad(locationLon), np.deg2rad(locationLat))
            mapped_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            print(f'POI: index x-y and lon-lat on model grid: {mapped_idx[1]}, {mapped_idx[0]}, {simLons[mapped_idx[0], mapped_idx[1]]}, {simLats[mapped_idx[0], mapped_idx[1]]}')

            # check if the POI is on a river, lake, or sea ParFlow grid element
            # if this is true, offer the user alternative grid elements
            # do not stop the procedure, perhaps a user deliberately wants to 
            # extract such a grid element
            if indicator[mapped_idx[0], mapped_idx[1]] in [19, 20, 21]:

                print("ATTENTION: your chosen POI (point of interest) is located directly on a lake, river, or ocean grid element, it is recommended to specify an alternative, nearby longitude-latitude coordinate pair")

                # find the indices that would sort the distances array in 
                # ascending order, i.e. grid elements with closest distance are
                # first; get the index tuples as x and y coordinate pairs of the 
                # dist array; get the indices of the closest grid elements to 
                # the original POI
                nr_of_neighbour_grid_pts = 10
                sorted_indices_2d = np.unravel_index(np.argsort(dist, axis=None), dist.shape)
                closest_indices = list(zip(sorted_indices_2d[0][1:nr_of_neighbour_grid_pts], sorted_indices_2d[1][1:nr_of_neighbour_grid_pts]))
                print("indices of alternative close-by grid elements to POI: ", closest_indices)

                # test if the alternative neighbouring grid elements are on a 
                # river, lake, or ocean grid element
                k = 0
                for j in range(len(closest_indices)):
                    if indicator[sorted_indices_2d[0][j], sorted_indices_2d[1][j]] not in [19, 20, 21]:
                        lon_f = "{:.5f}".format(simLons[sorted_indices_2d[0][j], sorted_indices_2d[1][j]])
                        lat_f = "{:.5f}".format(simLats[sorted_indices_2d[0][j], sorted_indices_2d[1][j]])
                        print("alternative, recommended coordinate pairs close-by, not on river, lake, or ocean, Lon-Lat [dec deg], 5 digits, edit your JASON file accordingly: ", lon_f, lat_f )
                        k += 1
                if k == 0:
                    print("there are no alterntive grid elements around your original POI which are NOT on a river, lake, or ocean grid element")

            # even if the given location is on a lake, river or ocean grid
            # point: get the data
            # assume the last variable is the actual data variable, this 
            # depends on the netCDF file structure and may need to be 
            # changed, it is a limitation
            # for some ensemble datasets there are several variables in the 
            # datafile
            # if you know the variable name, you may also directly set the
            # variable name here
            # before this does anything, check whether the file actually still
            # exists on the THREDDS server
            with Dataset(locationData, 'r') as ncData:
                
                variable = list(ncData.variables.keys())[-1]
                print("variable name: ", variable)
                print("variable long name: ", ncData.variables[variable].long_name)
                print("variable unit: ", ncData.variables[variable].units)
                print("variable dimension: ", ncData.variables[variable].ndim)
    
                time_var = ncData.variables['time']
                time_data = num2date(time_var[:], time_var.units, time_var.calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                #print(time_data)
                startDate = time_data[0].strftime("%Y%m%d")
                endDate = time_data[-1].strftime("%Y%m%d")
    
                if ncData.variables[variable].ndim == 3:
                    print(f'{ncData.variables[variable].long_name} is a 2D variable, no depth information needed')
                    var = ncData.variables[variable][:, mapped_idx[0], mapped_idx[1]]
                    depth_idx = 0
                else:
                    print(f'{ncData.variables[variable].long_name} is a 3D variable, extraction at specific depth')
                    # determine the vertical coordinate and extract timeseries
                    # the depth is in [m]
                    soil_layer_bnds = ncData.variables['soil_layer_bnds'][:, 0]
                    depth_idx, lower_boundary, upper_boundary = find_depth_index(abs(locationDepth), soil_layer_bnds)
                    var = ncData.variables[variable][:, depth_idx, mapped_idx[0], mapped_idx[1]]

                print("timeseries data at location and depth:", var)
    
                # if a CSV file shall be written, do this per data record, i.e.
                # entry in the JSON file
                if output_format == "csv":

                    # proper csv filename 
                    # locationData contains the URL
                    nameCSV = f'siteData_{locationID}_{locationData.split("/")[-1].split(".")[0]}_{startDate}-{endDate}_didx{depth_idx}.csv'
                    print(nameCSV)

                    #open an CSV and save the time series
                    fCSV = open(nameCSV, 'w', newline='')
                    writer = csv.writer(fCSV, delimiter=';')
    
                    writer.writerow(['locationID:', f'{locationID}'])
                    writer.writerow(['locationLon:', f'{locationLon}'])
                    writer.writerow(['locationLat:', f'{locationLat}'])
                    writer.writerow(['locationDepth:', f'{locationDepth}'])
                    writer.writerow(['variable:', ncData.variables[variable].long_name])
                    writer.writerow(['unit:', ncData.variables[variable].units])
                    writer.writerow(['data provider:', ncData.getncattr('institution')])
                    writer.writerow(['data source: ', 'https://doi.org/10.26165/JUELICH-DATA/GROHKP'])
                    writer.writerow(['data server: ', 'https://datapub.fz-juelich.de/slts/FZJ_ParFlow_DE06_hydrologic_forecasts/index.html'])
                    writer.writerow(['data license: ', 'CC BY 4.0'])
                    writer.writerow(['data extracted on:', datetime.datetime.today().strftime('%Y-%m-%d')])
                    writer.writerow(['temporal aggregation:', ncData.variables[variable].cell_methods])
                    # 2D variables are, e.g., wtd, tet, ifl
                    if ncData.variables[variable].ndim == 3:
                        writer.writerow(['depth: this is a 2D variable, no depth information'])
                    else:
                        # 3D variable with flux at a lower boundary
                        if variable == 'vsf':
                            writer.writerow(['depth:', f'{lower_boundary} m'])
                        # 3D variable with value representative for a depth layer
                        else:
                            writer.writerow(['depth (mid-point of soil layer):', f'{(lower_boundary+upper_boundary)/2.} m'])
    
                    for dd in range(len(time_data)):
                        #dayDate = time_data[dd].replace(hour=12, minute=0, second=0)
                        dayDate = time_data[dd].strftime("%Y-%m-%d")
                        var1Drow = [f'{dayDate}']
                        var1Drow.append(var[dd])
                        writer.writerow(var1Drow)
    
                    fCSV.close()
    
                # if this is the first JSON record, create 1D array to later on 
                # appand to the new array
                if i == 0:
                    var_array = np.array(var.reshape(1, var.size))
                else:
                    var_array = np.append(var_array, var.reshape(1, var.size), axis=0)
                #print(var_array.shape) 
                #print(var_array)

            i += 1

    print("----------------------------------------") 

    # data is always passed to the caller
    return(var_array)