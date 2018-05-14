from res.util import *

# make a calculator function
def ghiScore(x): return linearTransition( np.array(x), 3, 4 )
def roadDistScore(x): return linearTransition( np.array(x), 1000, 10000, True )
def powerScore(x): return linearTransition( np.array(x), 1000, 10000, True )
def settlementScore(x): return linearTransition( np.array(x), 500, 1000 )
def scoreOpenfieldPVLocation(ghi, roadDist, powerDist, settlementDist):
    return 0.5*ghiScore(ghi) + 0.2*roadDistScore(roadDist) + 0.2*powerScore(powerDist) + 0.1*settlementScore(settlementDist)
