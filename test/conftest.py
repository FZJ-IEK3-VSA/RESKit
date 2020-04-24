import pandas as pd

import reskit as rk
import pytest
# from reskit.workflows.wind import WindWorkflowGenerator


@pytest.fixture
def placements():
    placements = pd.read_csv(rk.TEST_DATA['turbine_placements.csv'])
    return placements


# @pytest.fixture
# def wind_workflow(placements):
#     wind_workflow = WindWorkflowGenerator(placements)
#     return wind_workflow
