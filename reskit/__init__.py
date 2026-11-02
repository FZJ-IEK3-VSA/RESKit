from reskit import cooling_heating, csp, dac, geothermal, solar, util, weather, wind
from reskit.parameters.parameters import OffshoreParameters, OnshoreParameters
from reskit.workflow_manager import (
    WorkflowManager,
    WorkflowQueue,
    distribute_workflow,
    execute_workflow_iteratively,
    load_workflow_result,
)
from reskit.weather.era5_source.era5_prepare import prepare_era5
from reskit.util.input_preparation import download_and_process

# The pre-0.6.0 CamelCase module paths, kept importable until RESKit 1.0.0.
from reskit import _deprecated_modules as _deprecated_modules

_deprecated_modules.install()


def __getattr__(name):
    """``reskit.TEST_DATA``, deprecated until RESKit 1.0.0, read through ETHOS.Data on access (PEP 562)."""
    if name == "TEST_DATA":
        from reskit.data import legacy_test_data

        return legacy_test_data()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
