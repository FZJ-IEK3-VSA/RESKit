from pathlib import Path

# create a DATA variable for solar which contains all data files in the solar module in a dict, based on filename as key
_data_folder = Path(__file__).parent / 'data'
DATA = {file.name: file for file in _data_folder.iterdir() if file.is_file()}

from .core.frank_correction import frank_correction_factors
from .core.system_design import location_to_module_tilt, location_to_module_azimuth, location_to_tracker_axis_tilt, location_to_tracker_axis_azimuth, location_to_cross_axis_tilt

from .workflows.solar_workflow_manager import SolarWorkflowManager

from .workflows.workflows import (
    openfield_pv_merra_ryberg2019,
    openfield_pv_era5,
    pv_era5_WinklerUnpublished,
    openfield_pv_sarah_unvalidated,
    openfield_pv_iconlam,
    openfield_pv_era5pure,
    openfield_pv_era5_unvalidated,
)
