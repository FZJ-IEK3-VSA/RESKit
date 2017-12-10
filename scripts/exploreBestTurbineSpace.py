from res.windpower import determineBestTurbine
import numpy as np
from multiprocessing import Pool
from datetime import datetime as dt

determineBestTurbine(roughness=0.01, weibK=2,weibL=6)

if __name__ == "__main__":
    kVals = np.linspace(1,3,21)[1:]
    roughnesses = np.logspace(-4,0.3,20)
    lVals = np.linspace(1,15,100)

    CPUS = 7
    pool = Pool(CPUS)
    startTime = dt.now()
    print("Starting at", startTime)
    results = []
    for r in roughnesses:
        for k in kVals:
            for l in lVals:
                inputs = dict(roughness=r,weibK=k,weibL=l)
                results.append( (inputs, pool.apply_async(determineBestTurbine, (), inputs)) )


    N = kVals.size*roughnesses.size*lVals.size
    count = -1
    with open("output.csv","w") as fo:
        fo.write("roughness,wibull-k,weibull-lambda,capacity,rotordiam,hubHeight\n")

        for inputs,res in results:
            count += 1
            if count%100==0: 
                print( "%.2f%% at +%ds"%(count/N*100, (dt.now()-startTime).total_seconds()))
            
            try:
                tmp = res.get()
                cap = tmp.capacity
                rd = tmp.rotordiam
                hh = tmp.hubHeight
            except:
                cap = -1
                rd = -1
                hh = -1

            fo.write("%f,%f,%f,%f,%f,%f\n"%( inputs["roughness"], inputs["weibK"], inputs["weibL"], cap, rd, hh ))


    pool.close()
    pool.join()