import geokit as gk
import pandas as pd
import pytest
from numpy import allclose

import reskit as rk
from reskit.geothermal.data.gringarten import Gringarten

# Default variables
sourceTemperature = rk.geothermal.data.path_temperatures
sourceSustainableHeatflow = rk.geothermal.data.path_heat_flow_sustainable_W_per_m2


@pytest.fixture
def placements():
    """Creates a DataFrame of geothermal placements with geometries."""
    df = pd.DataFrame()
    df["lat"] = [51.00, 37.0, 64.922, 0.0]
    df["lon"] = [9.00, -114.0, -18.854, 114.0]

    geoms = [gk.geom.point(lon, lat, srs=gk.srs.loadSRS(4326)) for lon, lat in zip(df["lon"], df["lat"])]
    df["geom"] = geoms

    return df


@pytest.fixture
def egs_output(placements):
    """Runs the egs_workflow and returns the output."""
    return rk.geothermal.egs_workflow(
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
    expected = [51.81435227, 28.62905446, 7.63056424, 20.2406851]
    assert allclose(egs_output.LCOE_SU_EUR_per_kWh, expected), "LCOE_SU values do not match!"


@pytest.fixture
def egs_output_triplette(placements):
    """Runs the egs_workflow for the triplette configuration and returns the output."""
    return rk.geothermal.egs_workflow(
        placements=placements,
        sourceTemperature=sourceTemperature,
        sourceSustainableHeatflow=sourceSustainableHeatflow,
        configuration="triplette",
        manual_values={"x_ED_1": 8},
        savepath=None,
    )


def test_gr_heat_flow_matches_pumped_flow_triplette(egs_output_triplette):
    """The flow the Gringarten model heats must equal the flow that is pumped and paid for."""
    out = egs_output_triplette
    T_inj = 80  # degC, default injection temperature of the plant data
    mdot_from_heat = out.Qdot_out_GR_MW * 1e6 / (Gringarten.cp_water * (out.T_Water_out_GR_degC - T_inj))
    assert allclose(mdot_from_heat, out.mdot_water_GR_kg_per_s), "Heated flow and pumped flow differ!"


def test_lcoe_triplette(egs_output_triplette):
    """Validate the LCOE values of all methods for the triplette configuration."""
    out = egs_output_triplette
    assert allclose(out.LCOE_VM_EUR_per_kWh, [0.78743612, 0.4884524, 0.23929582, 0.46798273])
    assert allclose(out.LCOE_GR_EUR_per_kWh, [0.29954464, 0.10733087, 0.03224863, 0.09617741])
    assert allclose(out.LCOE_SU_EUR_per_kWh, [49.27727635, 26.48838953, 6.7233954, 18.72754838])
