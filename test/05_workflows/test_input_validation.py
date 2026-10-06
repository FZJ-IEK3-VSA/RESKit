import importlib
import inspect

import geokit as gk
import numpy as np
import pandas as pd
import pytest

from reskit import TEST_DATA, data, validate_inputs
from reskit.util.input_validation import WORKFLOW_FAMILIES, _weather_reads

WORKFLOW = "wind_era5_PenaSanchezDunkelWinklerEtAl2025"


@pytest.fixture
def wind_inputs():
    placements = gk.vector.extractFeatures(TEST_DATA["turbinePlacements.shp"])
    placements["hub_height"] = 120
    placements["capacity"] = 3000
    placements["rotor_diam"] = 150
    inputs = data.paths(WORKFLOW, test=True)
    arguments = dict(
        era5_path=inputs["era5"],
        gwa_100m_path=inputs["gwa_100m"],
        height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
    )
    return placements, arguments


def test_validate_inputs_accepts_valid_inputs(wind_inputs):
    placements, arguments = wind_inputs

    report = validate_inputs(WORKFLOW, placements, **arguments)

    assert report.ok, report
    assert not report.warnings, report


def test_validate_inputs_reports_every_problem(wind_inputs, tmp_path):
    placements, arguments = wind_inputs
    placements = placements.drop(columns="hub_height")
    far_away = pd.DataFrame({"lon": [20.0], "lat": [60.0], "capacity": [3000], "rotor_diam": [150]})
    placements = pd.concat([placements.drop(columns="geom").assign(lon=6.1, lat=50.6), far_away])
    arguments["gwa_100m_path"] = str(tmp_path / "missing.tif")

    report = validate_inputs(WORKFLOW, placements, output_netcdf_path=str(tmp_path / "no" / "out.nc"), **arguments)

    errors = {(finding.check, finding.message.split(":")[0]) for finding in report.errors}
    assert ("placements", "need the column 'hub_height'") in errors
    assert ("gwa_100m_path", "does not exist") in errors
    assert ("output_netcdf_path", "the output directory does not exist") in errors
    assert any(check == "weather 'era5_path'" and "does not cover 1 of" in m for check, m in errors)
    assert {finding.check for finding in report.warnings} == {"height_scaling_data[50]", "height_scaling_data[200]"}


def test_validate_inputs_reports_unavailable_weather_variable():
    placements = pd.DataFrame(
        {"lon": [6.1], "lat": [50.6], "capacity": [3000], "hub_height": [120], "rotor_diam": [150]}
    )

    # MERRA provides no boundary layer height, which wind_config reads
    report = validate_inputs(
        "wind_config", placements, weather_path=TEST_DATA["merra-like"], weather_source_type="MERRA"
    )

    weather_errors = [f.message for f in report.errors if f.check == "weather 'weather_path'"]
    assert len(weather_errors) == 1 and "cannot provide 'boundary_layer_height'" in weather_errors[0]


def _weather_workflows():
    for family in WORKFLOW_FAMILIES:
        module = importlib.import_module(f"reskit.{family}.workflows.workflows")
        for name, function in inspect.getmembers(module, inspect.isfunction):
            source = inspect.getsource(function)
            if (
                function.__module__ == module.__name__
                and "path" in str(inspect.signature(function))
                and ".read(" in source
            ):
                yield function


@pytest.mark.parametrize("workflow", list(_weather_workflows()), ids=lambda f: f.__name__)
def test_validate_inputs_finds_the_weather_source_of_every_workflow(workflow):
    assert _weather_reads(workflow)
