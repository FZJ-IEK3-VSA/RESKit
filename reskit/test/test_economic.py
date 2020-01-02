from reskit.economic import *
import numpy as np

def test_lcoeSimple():
    # Test single
    capex = 1000
    meanProduction = 200
    opexPerCapex = 0.05
    lifetime = 20
    discountRate = 0.08

    a = lcoeSimple(capex=capex, meanProduction=meanProduction, opexPerCapex=opexPerCapex, 
                   lifetime=lifetime, discountRate=discountRate)
    if not np.isclose(a, 0.759261044116): raise RuntimeError( "Single evaluation" )


    # Test multiple
    capex = np.array([1000, 1500, 2000,])
    meanProduction = np.array([200, 150, 100])
    opexPerCapex = 0.05
    lifetime = 20
    discountRate = 0.08

    a = lcoeSimple(capex=capex, meanProduction=meanProduction, opexPerCapex=opexPerCapex, 
                   lifetime=lifetime, discountRate=discountRate)
    if not np.isclose(a, [ 0.75926104,  1.51852209,  3.03704418]).all(): 
        raise RuntimeError( "Multiple evaluation" )

def test_lcoe():
    expenditures = np.array([10000, 1000, 1000, 1000, 1000, 1000, 1000])
    productions = np.array([5000,4000,2000,4000,3000,5000,4000])
    discountRate = 0.08

    a = lcoe( expenditures=expenditures, productions=productions, discountRate =discountRate )
    if not np.isclose(a, 0.673170711385): raise RuntimeError( "Single evaluation" )

if __name__ == "__main__":
    test_lcoeSimple()
    test_lcoe()