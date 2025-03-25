from pathlib import Path
from . import gringarten

# Get the current directory of the 'data' package
DATA_DIR = Path(__file__).resolve().parent

# Define file paths
path_heat_flow_sustainable_W_per_m2 = DATA_DIR / "heat_flow_sustainable_W_per_m2.nc4"  # Replace with actual filename
path_temperatures = DATA_DIR / "Temperatures.nc4"  # Replace with actual filename
