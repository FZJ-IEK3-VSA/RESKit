from pathlib import Path

from reskit.geothermal.data.gringarten import Gringarten

# Get the current directory of the 'data' package
DATA_DIR = Path(__file__).resolve().parent

# Define file paths
path_heat_flow_sustainable_w_per_m2 = DATA_DIR / "heat_flow_sustainable_W_per_m2.nc4"  # Replace with actual filename
path_temperatures = DATA_DIR / "Temperatures.nc4"  # Replace with actual filename
