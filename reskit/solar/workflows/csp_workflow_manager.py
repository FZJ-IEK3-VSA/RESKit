from ...workflow_manager import WorkflowManager
import numpy as np

class CSPWorkflowManager(WorkflowManager):
    def __init__(self, placements):
        """

        __init_(self, placements)

        Initialization of an instance of the generic SolarWorkflowManager class.

        Parameters
        ----------
        placements : pandas Dataframe
                     The locations that the simulation should be run for.
                     Columns must include "lon", "lat"

        Returns
        -------
        SolarWorkflorManager

        """

        # Do basic workflow construction
        super().__init__(placements)
        self._time_sel_ = None
        self._time_index_ = None
        self.module = None

    def easycalc(self):
        area_usage = 0.5
        geometric_efficiency = 0.7
        rankine_eficency = 0.3

        self.sim_data['Power'] = np.multiply(self.placements['Area'].to_numpy(), self.sim_data['direct_horizontal_irradiance']) * area_usage * geometric_efficiency * rankine_eficency

        return self
