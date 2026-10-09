import hashlib
from importlib import metadata

import numpy as np
import pandas as pd
import xarray as xr

import reskit as rk
from reskit.util.provenance import read_provenance, record_provenance


@record_provenance
def _workflow(placements, correction_path, factor=2.0, output_netcdf_path=None):
    wf = rk.WorkflowManager(placements)
    wf.set_time_index(pd.date_range("2020-01-01", periods=3, freq="h"))
    wf.sim_data["capacity_factor"] = np.full((3, len(placements)), 0.5 * factor)
    wf.record_input_file("cf_correction", correction_path)
    return wf.to_xarray(output_netcdf_path=output_netcdf_path)


def test_workflow_results_record_their_provenance(tmp_path):
    correction = tmp_path / "correction.csv"
    correction.write_bytes(b"1.0\n")
    placements = pd.DataFrame({"lon": [6.0, 7.0], "lat": [50.0, 51.0]})

    output = tmp_path / "result.nc"
    output.write_text("an earlier result, which is no input")
    _workflow(placements, correction, output_netcdf_path=str(output))
    provenance = read_provenance(xr.load_dataset(output))

    assert provenance["version"] == metadata.version("reskit")
    assert provenance["workflow"].endswith("._workflow")
    assert provenance["crs"] == "EPSG:4326"
    assert "numpy" in provenance["dependencies"]
    # the placements are part of the dataset, the other arguments are recorded
    assert provenance["parameters"] == {
        "correction_path": str(correction),
        "factor": 2.0,
        "output_netcdf_path": str(output),
    }
    (record,) = provenance["input_files"]
    assert record["path"] == str(correction)
    assert record["roles"] == ["argument:correction_path", "cf_correction"]
    assert record["sha256"] == hashlib.sha256(b"1.0\n").hexdigest()
