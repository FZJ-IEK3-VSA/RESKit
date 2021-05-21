import pytest
from papermill import execute_notebook
from glob import glob
from os import path

TEST_DIR = path.dirname(__file__)
notebooks = glob(path.join(TEST_DIR, "..", "..", "examples", "*.ipynb"))

# This will create a test function for each ".ipynb" file in the examples folder
# Note: The notebooks are only checked that they can run without rasing an error.
#       The outputs are not tested

@pytest.mark.skip(reason="Skip for now. Better notebook testing will be implemented.")
for notebook in notebooks:
    notebook_name = path.splitext(path.basename(notebook))[0]
    globals()["test_" + notebook_name] = lambda: execute_notebook(notebook, notebook_name + ".ipynb")
