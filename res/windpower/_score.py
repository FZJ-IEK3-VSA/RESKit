from ._util import *

# make a calculator function
def scoreOnshoreWindLocation(windspeed, roadDist, powerDist, settlementDist,
                             windspeedStart=4.5, roadDistStart=1000, powerDistStart=1000, settlementDistStart=500,
                             windspeedStop=7, roadDistStop=10000, powerDistStop=10000, settlementDistStop=1000,
                             windspeedFlip=False, roadDistFlip=True, powerDistFlip=True, settlementDistFlip=False,
                             windspeedWeight=0.5, roadDistWeight=0.2, powerDistWeight=0.2, settlementDistWeight=0.1,)
    
    totalScore = windspeedWeight * linearTransition( np.array(windspeed), windspeedStart, windspeedStop , windspeedFlip)
    totalScore += roadDistWeight * linearTransition( np.array(roadDist), roadDistStart, roadDistStop , roadDistFlip)
    totalScore += powerDistWeight * linearTransition( np.array(powerDist), powerDistStart, powerDistStop , powerDistFlip)
    totalScore += settlementDistWeight * linearTransition( np.array(settlementDist), settlementDistStart, settlementDistStop , settlementDistFlip)
    return totalScore
