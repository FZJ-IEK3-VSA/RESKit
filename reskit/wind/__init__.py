from importlib import import_module

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

# The two cost models are resolved on first use rather than on import. They pull in
# WISDEM, which costs ~1.9 s and ~390 MB to load, and nothing in the simulation
# workflows touches them -- so a worker which only simulates paid for a cost model it
# never called. Accessing either name still works exactly as before, it just loads
# WISDEM at that point instead of at 'import reskit'.
_LAZY_IMPORTS = {
    "onshore_turbine_capex": ".economic.onshore_cost_model",
    # calculateSpecificOffshoreCapex is the deprecated alias (#226), removed in v1.0.0
    "calculateSpecificOffshoreCapex": ".economic.offshore_cost_model",
    "calculate_specific_offshore_capex": ".economic.offshore_cost_model",
}


def __getattr__(name):
    """Import the cost models on first attribute access (PEP 562)."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(_LAZY_IMPORTS[name], __name__), name)
    globals()[name] = value  # bind it, so the lookup only happens once
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


from .workflows.wind_workflow_manager import WindWorkflowManager
from .workflows.workflows import *
