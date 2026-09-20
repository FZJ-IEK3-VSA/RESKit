from . import workflows
from . import data
from . import preprocessing
from .preprocessing import create_geothermal_resource

# EGSworkflow is the deprecated alias of egs_workflow (#226), removed in v1.0.0
from .workflows.workflows import EGSworkflow, egs_workflow
