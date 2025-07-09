# %%
import pandas as pd
import numpy as np
import os
import pickle
import glob
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

# %%
WaterdepthFolderPath="/benchtop/shared_data/General_Bathymetric_Chart_of_the_Oceans_GEBCO/GEBCO_2020/GEBCO_tiles/"
tif_files = glob.glob(os.path.join(WaterdepthFolderPath, "*.tif"))

#load the relevant distance file beforehand to safe time (Huge fileH 1.3 GB)
DistancetoShorePath="/benchtop/projects/2021-m-stargardt-phd/02_GHR_2025/01_offshoreTiffs/GMT_intermediate_coast_distance_01d.tif"

with rasterio.open(DistancetoShorePath) as src:

    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    # Read the raster band once
    band = src.read(1)

# %%
# TODO. we need to get lat and lon location data vor turibine and thereby receive depth values 
lon=row['lon']
lat=row['lat']
result_depth = get_raster_value_from_tifs(tif_files, lat, lon)
print(result_depth)

# get relevant information about distance to coastine
x, y = transformer.transform(lon, lat)

try:
    row, col = rowcol(src.transform, x, y)

    0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
    value = band[row, col]


except Exception as e:
    print(f"Error at Lat: {lat}, Lon: {lon}: {e}")

# %%
def calculateOffshoreCapex(InputCapex,capacity,hubheight,waterdepth,coastDistance,rotordiam,shareTurb=0.3,shareFound=0.3,shareCable=0.3, shareOverhead=0.1,maxMonopolDepth=25,maxJacketDepth=55):

    "thus function scales the geenric Offshore Capex regarding watrdepth and distance to the nearest coastline"


    assert np.isclose(shareTurb + shareFound + shareCable + shareOverhead,1.0,rtol=1e-9) , "Sum of all cost shares must equal 1"
    assert 0<maxMonopolDepth<60,'Maximum Depth for Monopile Foundation must be between 0 and 50 m'
    assert 60<=maxJacketDepth<100,'Maximum Depth for Jacket Foundation must be between 0 and 50 m'
    assert maxMonopolDepth<maxJacketDepth, ' Maximum Depth for Jacket Foundation must be larger than maximum  depth for Monopile Foundation'

    averageDepthLiterature=17 # m 
    averageCoastDistance=27 # km 

    TurbineCostBase=InputCapex*shareTurb
    FoundCostbase=InputCapex*shareFound
    CableCostBase=InputCapex*shareCable
    OverheadCostBase=InputCapex*shareOverhead
    

    # adapting each cost share 
    # Turbine cost are adapted to Severin's onshore cost approach 

    #ToDo
 
    # 9.7 MW capacity as standard (see literautre) hubheight=137m rotor diam=216m
    baseCap=9.7 
    baseHubHeight=137 # literature
    baserotorDiameter=216 # literature

    #Scaling new turbines cost according to their dimesniosn and acccording to severins calculations
    TurbineCostNew=onshore_tcc(capacity, hubheight, rotordiam, gdp_escalator=1, blade_material_escalator=1, blades=3)
    TurbineCostRefernce=onshore_tcc(baseCap,baseHubHeight,baserotorDiameter, gdp_escalator=1, blade_material_escalator=1, blades=3)

    costRatioTurbine=TurbineCostNew/TurbineCostRefernce
    NewTurbineCost=TurbineCostBase*costRatioTurbine


    #Found Cost Base 
    # Adapting the New foundation costs 
    DeptBaseCost=getRatedCostfromWaterdepth(averageDepthLiterature) #base depth of CAPEX
    DeptPlantCost=getRatedCostfromWaterdepth(waterdepth) # depth of calcualted power plant 
    CostRatio=DeptPlantCost/DeptBaseCost
    NewFoundationCost=FoundCostbase*CostRatio

    # Cable cost and connection cost  
    # applicaton of cost for DC connection from power plant to coast as scaling factor 
    
    #
    ratioCable=getCableCost(coastDistance)/getCableCost(averageCoastDistance)

    NewCableCost=CableCostBase*ratioCable


    #New Capex

    TotalOffshoreCapEx= NewTurbineCost+NewFoundationCost+NewCableCost+OverheadCostBase

    return TotalOffshoreCapEx


# %%
def get_raster_value_from_tifs(tif_paths, latitude, longitude):
    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds  # left, bottom, right, top
            
            # Check if the point is inside this raster
            if (bounds.left <= longitude <= bounds.right) and (bounds.bottom <= latitude <= bounds.top):
                try:
                    # Use safer sampling method
                    for val in src.sample([(longitude, latitude)]):
                        
                        return val[0]
                        
                except Exception as e:
                    print(f"Error reading from {tif_path}: {e}")
                    continue
    return None  # Not found in any tile

# %%
def getRatedCostfromWaterdepth(Depth,allowNegative=True):
    # based on Rogeau et al (2023), Doi: 10.1016/j.rser.2023.113699
    if (not allowNegative) and Depth<0:
        raise ValueError('Depth must not be negative when not allowNegative')
    if  Depth <25:
        c1=181
        c2=552
        c3=370    
    elif Depth >=25 and Depth <=55:
        c1=103
        c2=-2043
        c3=478
    else:
        c1=0
        c2=697
        c3=1223

    
    RatedCost=c1*(Depth**2)+c2*Depth+c3*1000

    return RatedCost

# %%
def getCableCost(distance, capacity):

    """A function to get the cost for connecting a off shore windpower plant to the coastline according to Rogeau et al (2023), Doi: 10.1016/j.rser.2023.113699

    Parameters
    ----------
    distance :  float
                distance to caostline in km 
    capacity :  float
                powerplant's capacity in MW

    """

    FixeCost=0
    
    variableCost= 1.35 *distance*(capacity*1e6)
    cableCost=FixeCost+variableCost

    return cableCost


# %%
def onshore_tcc(
    cp, hh, rd, gdp_escalator=1, blade_material_escalator=1, blades=3
):
    """
    A function to determine the turbine capital cost (TCC) of a 3 blade standar onshore wind turbine based capacity, hub height and rotor diameter values according to the cost model by Fingersh et al. [1].

    Parameters
    ----------
    cp : numeric or array-like
        Turbine's capacity in kW
    hh : numeric or array-like
        Turbine's hub height in m
    rd : numeric or array-like
        Turbine's rotor diamter in m
    gdp_escalator : int, optional
        Labor cost escalator, by default 1
    blade_material_escalator : int, optional
        Blade material cost escalator, by default 1
    blades : int, optional
        Number of blades, by default 3

    Returns
    -------
    numeric or array-like
        Turbine's turbine capital cost (TCC) in monetary units.

    References
    ---------
    [1] Fingersh, L., Hand, M., & Laxson, A. (2006). Wind Turbine Design Cost and Scaling Model. NREL. https://www.nrel.gov/docs/fy07osti/40566.pdf

    """
    # initialize OnshoreParameters class and feed with custom param values


    rr = rd / 2
    sa = np.pi * rr * rr

    # Blade Cost
    singleBladeMass = 0.4948 * np.power(rr, 2.53)
    singleBladeCost = (
        (0.4019 * np.power(rr, 3) - 21051) * blade_material_escalator
        + 2.7445 * np.power(rr, 2.5025) * gdp_escalator
    ) * (1 - 0.28)

    # Hub
    hubMass = 0.945 * singleBladeMass + 5680.3
    hubCost = hubMass * 4.25

    # Pitch and bearings
    # pitchBearingMass = 0.1295 * (singleBladeMass * blades) + 491.31
    # pitchSystemMass = pitchBearingMass*1.328+555
    pitchSystemCost = 2.28 * (0.2106 * np.power(rd, 2.6578))

    # Spinner and nosecone
    noseConeMass = 18.5 * rd - 520.5
    noseConeCost = noseConeMass * 5.57

    # Low Speed Shaft
    # lowSpeedShaftMass = 0.0142 * np.power(rd, 2.888)
    lowSpeedShaftCost = 0.01 * np.power(rd, 2.887)

    # Main bearings
    bearingMass = (rd * 8 / 600 - 0.033) * 0.0092 * np.power(rd, 2.5)
    bearingCost = 2 * bearingMass * 17.6

    # Gearbox
    # Gearbox not included for direct drive turbines

    # Break, coupling, and others
    breakCouplingCost = 1.9894 * cp - 0.1141
    # breakCouplingMass = breakCouplingCost/10

    # Generator (Assuming direct drive)
    # generatorMass = 6661.25 * np.power(lowSpeedShaftTorque, 0.606) # wtf is the torque?
    generatorCost = cp * 219.33

    # Electronics
    electronicsCost = cp * 79

    # Yaw drive and bearing
    # yawSystemMass = 1.6*(0.0009*np.power(rd, 3.314))
    yawSystemCost = 2 * (0.0339 * np.power(rd, 2.964))

    # Mainframe (Assume direct drive)
    mainframeMass = 1.228 * np.power(rd, 1.953)
    mainframeCost = 627.28 * np.power(rd, 0.85)

    # Platform and railings
    platformAndRailingMass = 0.125 * mainframeMass
    platformAndRailingCost = platformAndRailingMass * 8.7

    # Electrical Connections
    electricalConnectionCost = cp * 40

    # Hydraulic and Cooling systems
    # hydraulicAndCoolingSystemMass = 0.08 * cp
    hydraulicAndCoolingSystemCost = cp * 12

    # Nacelle Cover
    nacelleCost = 11.537 * cp + 3849.7
    # nacelleMass = nacelleCost/10

    # Tower
    towerMass = 0.2694 * sa * hh + 1779
    towerCost = towerMass * 1.5

    # Add up the turbine capital cost
    turbineCapitalCost = (
        singleBladeCost * blades
        + hubCost
        + pitchSystemCost
        + noseConeCost
        + lowSpeedShaftCost
        + bearingCost
        + breakCouplingCost
        + generatorCost
        + electronicsCost
        + yawSystemCost
        + mainframeCost
        + platformAndRailingCost
        + electricalConnectionCost
        + hydraulicAndCoolingSystemCost
        + nacelleCost
        + towerCost
    )

    return turbineCapitalCost


