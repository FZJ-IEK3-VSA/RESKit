from res.windpower import determineBestTurbine
import numpy as np
from multiprocessing import Pool

determineBestTurbine(roughness=0.01, weibK=2,weibL=6)

if __name__ == "__main__":
    kVals = np.linspace(1,3,51)[1:]
    roughnesses = np.logspace(-4,0.3,50)
    lVals = np.linspace(1,15,151)

    CPUS = 7
    pool = Pool(CPUS)

    results = []
    for r in roughnesses:
        for k in kVals:
            for l in lVals:
                inputs = dict(roughness=r,weibK=k,weibL=l)
                results.append( (inputs, pool.apply_async(determineBestTurbine, (), inputs)) )


    N = kVals.size*roughnesses.size*lVals.size
    count = -1
    with open("output.csv","w") as fo:
        count += 1
        if count%100==0: print(count/N*100)
        fo.write("roughness,wibull-k,weibull-lambda,capacity,rotordiam,hubHeight\n")

        for inputs,res in results:
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