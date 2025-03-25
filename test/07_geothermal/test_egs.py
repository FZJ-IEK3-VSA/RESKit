import pytest
import reskit as rk
import pandas as pd
import geokit as gk
from numpy import allclose

# Default variables
sourceTemperature = rk.geothermal.data.path_temperatures
sourceSustainableHeatflow = rk.geothermal.data.path_heat_flow_sustainable_W_per_m2

@pytest.fixture
def placements():
    """Creates a DataFrame of geothermal placements with geometries."""
    df = pd.DataFrame()
    df['lat'] = [51.00, 37.0, 64.922, 0.0]
    df['lon'] = [9.00, -114.0, -18.854, 114.0]

    geoms = [gk.geom.point(lon, lat, srs=gk.srs.loadSRS(4326)) for lon, lat in zip(df['lon'], df['lat'])]
    df['geom'] = geoms

    return df

@pytest.fixture
def egs_output(placements):
    """Runs the EGSworkflow and returns the output."""
    return rk.geothermal.EGSworkflow(
        placements=placements,
        sourceTemperature=sourceTemperature,
        sourceSustainableHeatflow=sourceSustainableHeatflow,
        manual_values={"x_ED_1": 8},
        savepath=None,
    )

def test_output_length(placements, egs_output):
    """Check that the number of placements in the output matches the input."""
    assert len(egs_output.placements) == len(placements), "Mismatch in number of placements."

def test_lcoe_vm(egs_output):
    """Validate LCOE_VM_EUR_per_kWh values."""
    expected = [0.83377206, 0.52975674, 0.26825743, 0.50673514]
    assert allclose(egs_output.LCOE_VM_EUR_per_kWh, expected), "LCOE_VM values do not match!"

def test_lcoe_gr(egs_output):
    """Validate LCOE_GR_EUR_per_kWh values."""
    expected = [0.32707465, 0.12043728, 0.03419219, 0.10778247]
    assert allclose(egs_output.LCOE_GR_EUR_per_kWh, expected), "LCOE_GR values do not match!"

def test_lcoe_su(egs_output):
    """Validate LCOE_SU_EUR_per_kWh values."""
    expected = [51.79376029, 28.61874858, 7.62909725, 20.23294058]
    assert allclose(egs_output.LCOE_SU_EUR_per_kWh, expected), "LCOE_SU values do not match!"
