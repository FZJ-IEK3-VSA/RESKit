# import geokit as gk
import reskit as rk
from reskit import windpower

# import pandas as pd
# import numpy as np
# from os import mkdir, environ
# from os.path import join, isfile, isdir
# from collections import OrderedDict, namedtuple
# from types import FunctionType
from .wind_workflow_generator import WindWorkflowGenerator


def onshore_wind_Ryberg2019_Europe(placements, merra_path, gwa_50m_path, clc2012_path):
    wf = WindWorkflowGenerator(placements)

    wf.read(
        variables=['elevated_wind_speed',
                   "surface_pressure",
                   "surface_air_temperature"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True)

    wf.adjust_variable_to_long_run_average(
        variable='windspeed_for_wind_energy',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_WINDSPEED,
        real_long_run_average=gwa_50m_path
    )

    wf.estimate_roughness_from_land_coverestimate_roughness_from_land_cover(
        path=clc2012_path,
        source_type="clc")

    wf.logarithmic_projection_of_wind_speeds_to_hub_height()

    wf.apply_air_density_correction_to_wind_speeds()

    wf.convolute_power_curves(
        stdScaling=0.06,
        stdBase=0.1
    )

    wf.simulate()

    wf.apply_loss_factor(
        loss=lambda x: windpower.lowGenCorrection(x, base=0.0, sharpness=5.0)
    )

    onshore_capacity_factor = wf.sim_data['capacity_factor']

    return onshore_capacity_factor
