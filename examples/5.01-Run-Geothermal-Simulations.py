import reskit as rk

import pandas as pd
import geokit as gk
from numpy import allclose

# just for knowldege, they are default variables
sourceTemperature = rk.geothermal.data.path_temperatures
sourceSustainableHeatflow = rk.geothermal.data.path_heat_flow_sustainable_W_per_m2


placements = pd.DataFrame()
placements['lat'] = [51.00, 37.0, 64.922, 0.0]
placements['lon'] = [9.00, -114.0, -18.854, 114.0]

geoms =[]
for i in range(len(placements)):
    x = placements['lon'][i]
    y = placements['lat'][i]
    geom = gk.geom.point(x,y, srs=gk.srs.loadSRS(4326))
    geoms.append(geom)
placements['geom']= geoms



out_xed8 = rk.geothermal.EGSworkflow(
    placements = placements,
    sourceTemperature = sourceTemperature,
    sourceSustainableHeatflow=sourceSustainableHeatflow,
    manual_values={"x_ED_1": 8},
    savepath=None,
)

print(out_xed8)


assert len(out_xed8.placements) == len(placements)

expected = [0.83377206, 0.52975674, 0.26825743, 0.50673514]
assert allclose(out_xed8.LCOE_VM_EUR_per_kWh, expected), "Values do not match!"

expected = [0.32707465, 0.12043728, 0.03419219, 0.10778247]
assert allclose(out_xed8.LCOE_GR_EUR_per_kWh, expected), "Values do not match!"

expected = [51.79376029, 28.61874858,  7.62909725, 20.23294058]
assert allclose(out_xed8.LCOE_SU_EUR_per_kWh, expected), "Values do not match!"