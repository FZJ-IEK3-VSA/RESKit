from .core.air_density_adjustment import apply_air_density_adjustment
from .core.design_turbine import onshore_turbine_from_avg_wind_speed
from .core.logarithmic_profile import (
    apply_logarithmic_profile_projection,
    roughness_from_clc,
    roughness_from_land_cover_classification,
    roughness_from_land_cover_source,
    roughness_from_levels,
)
from .core.power_curve import PowerCurve, compute_specific_power
from .core.power_profile import alpha_from_levels, apply_power_profile_projection

# TurbineLibrary is the deprecated alias of turbine_library (#226), removed in v1.0.0
from .core.turbine_library import TurbineLibrary, turbine_library
from .core.design_turbine import turbine_design_from_avg_wind_speed, onshore_turbine_from_avg_wind_speed

from .economic.onshore_cost_model import onshore_turbine_capex

# calculateSpecificOffshoreCapex is the deprecated alias (#226), removed in v1.0.0
from .economic.offshore_cost_model import (
    calculateSpecificOffshoreCapex,
    calculate_specific_offshore_capex,
)


from .workflows.wind_workflow_manager import WindWorkflowManager
from .workflows.workflows import *
