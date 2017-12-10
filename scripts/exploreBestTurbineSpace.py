from res.windpower import determineBestTurbine
import numpy as np
from multiprocessing import Pool

roughnesses = np.logspace(-4,0.3,50)
kVals = np.linspace(1,3,51)[1:]
lVals = np.linspace(1,15,151)

CPUS = 4
pool = Pool(CPUS)

results = []
for r in roughnesses[:3]:
	for k in kVals[:3]:
		for l in lVals[:3]:
			inputs = dict(roughness=r,weibK=k,weibL=l)
			results.append( (inputs, pool.apply_async(determineBestTurbine, (), inputs)) )

pool.close()
pool.join()

with open("output.csv","w") as fo:
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

