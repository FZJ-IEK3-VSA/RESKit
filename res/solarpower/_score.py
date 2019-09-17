from res.util.util_ import *

# make a calculator function
def scoreOpenfieldPVLocation(ghi, roadDist, powerDist, settlementDist,
                             ghiStart=3, roadDistStart=1000, powerDistStart=1000, settlementDistStart=500,
                             ghiStop=4, roadDistStop=10000, powerDistStop=10000, settlementDistStop=1000,
                             ghiFlip=False, roadDistFlip=True, powerDistFlip=True, settlementDistFlip=False,
                             ghiWeight=0.5, roadDistWeight=0.2, powerDistWeight=0.2, settlementDistWeight=0.1,):
	"""Performs a multi-criteria scoring of potential open-field PV site based off:
        * Average global horizontal irradiance
        * Distance from the nearest settlement area
        * Distance from the nearest roadway
        * Distance from the nearest power line
    """
    totalScore = ghiWeight * linearTransition( np.array(ghi), ghiStart, ghiStop , ghiFlip)
    totalScore += roadDistWeight * linearTransition( np.array(roadDist), roadDistStart, roadDistStop , roadDistFlip)
    totalScore += powerDistWeight * linearTransition( np.array(powerDist), powerDistStart, powerDistStop , powerDistFlip)
    totalScore += settlementDistWeight * linearTransition( np.array(settlementDist), settlementDistStart, settlementDistStop , settlementDistFlip)
    return totalScore
