__version__ = "0.4.0"

from . import util
from . import weather
import yaml
from os.path import join, dirname

from .workflow_manager import (
    WorkflowManager,
    WorkflowQueue,
    distribute_workflow,
    load_workflow_result,
    execute_workflow_iteratively,
)
from . import wind
from . import solar
from . import csp

from ._test import TEST_DATA
from .parameters.parameters import OnshoreParameters, OffshoreParameters

DEFAULT_PATHS = {}
with open(join(dirname(__file__), "default_paths.yaml")) as f:
    DEFAULT_PATHS = yaml.safe_load(f)
