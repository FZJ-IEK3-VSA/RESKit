from ._util import *

# make a calculator function
def windspeedScore(x): return linearTransition( np.array(x), 4.5, 7 )
def roadDistScore(x): return linearTransition( np.array(x), 1000, 10000, True )
def powerScore(x): return linearTransition( np.array(x), 1000, 10000, True )
def settlementScore(x): return linearTransition( np.array(x), 500, 1000 )
def scoreOnshoreWindLocation(windspeed, roadDist, powerDist, settlementDist):
    return 0.5*windspeedScore(windspeed) + 0.2*roadDistScore(roadDist) + 0.2*powerScore(powerDist) + 0.1*settlementScore(settlementDist)
