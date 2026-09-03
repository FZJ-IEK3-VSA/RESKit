"""Execute the example notebooks with papermill.

The notebooks are only checked that they run without raising an error. The outputs are
not tested.

Every notebook test carries the "notebooks" marker. The marker is deselected by
default, because the notebooks need several minutes. Run them with:

    pytest -m notebooks
"""

import ast
import json
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = TEST_DIR.parent.parent / "examples"

# The notebooks live in sub-directories of the examples folder, therefore the glob is
# recursive. Checkpoint copies of Jupyter are not example notebooks.
NOTEBOOKS = sorted(
    notebook for notebook in EXAMPLES_DIR.glob("**/*.ipynb") if ".ipynb_checkpoints" not in notebook.parts
)

# The documentation build excludes the same notebooks, see DEFAULT_EXCLUDED and its
# comment in docs/scripts/execute_notebooks.py. They need credentials, a large download
# or a purchased dataset, or they are recipes with placeholder paths. The CI workflows
# leave out the same set, see the "examples_to_execute" input of the workflows in
# .github/workflows/.
DOCS_NOTEBOOK_SCRIPT = TEST_DIR.parent.parent / "docs" / "scripts" / "execute_notebooks.py"
NOTEBOOKS_WITHOUT_LOCAL_DATA = {
    "1_1_1_how_to_download_era5_data",
    "1_1_2_wind_speed_from_vectors_in_era5",
    "1_1_3_prepare_era5_for_wind_workflow",
    "1_1_4_prepare_era5_for_solar_workflow",
    "1_3_1_process_power_curves_from_thewindpower_net",
    "1_4_1_how_to_create_LRA_datasets",
}


def _notebook_id(notebook: Path) -> str:
    """Give the notebook path relative to the examples folder as the test id."""
    return notebook.relative_to(EXAMPLES_DIR).as_posix()


def _copy_without_skipped_cells(notebook: Path, target: Path) -> Path:
    """Copy the notebook without the cells which carry the NBVAL_SKIP marker.

    The CI executes the notebooks with nbval, which does not run a cell that contains
    "# NBVAL_SKIP". papermill executes every cell. Therefore these cells are removed
    here, and both runners execute the same cells.
    """
    content = json.loads(notebook.read_text(encoding="utf-8"))
    content["cells"] = [cell for cell in content["cells"] if "NBVAL_SKIP" not in "".join(cell["source"])]
    target.write_text(json.dumps(content), encoding="utf-8")
    return target


def _docs_excluded_notebooks() -> set:
    """Read DEFAULT_EXCLUDED from the documentation script, without an import."""
    tree = ast.parse(DOCS_NOTEBOOK_SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", [])
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULT_EXCLUDED" for t in targets):
            return {Path(name).stem for name in ast.literal_eval(node.value)}
    raise AssertionError(f"DEFAULT_EXCLUDED not found in {DOCS_NOTEBOOK_SCRIPT}")


def test_the_skip_list_agrees_with_the_documentation_build():
    """The skipped notebooks must be the notebooks which the docs build also excludes.

    The docs build uses nbclient, which does not honor the NBVAL_SKIP marker, therefore
    it excludes 3_8 as well. These tests remove the marked cells, therefore 3_8 runs.
    """
    assert NOTEBOOKS_WITHOUT_LOCAL_DATA == _docs_excluded_notebooks() - {"3_8_use_workflows_with_zarr"}


def test_the_example_notebooks_are_collected():
    """The notebook tests must not silently collect zero notebooks."""
    assert EXAMPLES_DIR.is_dir(), f"{EXAMPLES_DIR} is not a directory"
    assert NOTEBOOKS, f"no example notebook found below {EXAMPLES_DIR}"


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_notebook_id)
def test_example_notebook(notebook, tmp_path):
    """Execute one example notebook and check that it does not raise."""
    if notebook.stem in NOTEBOOKS_WITHOUT_LOCAL_DATA:
        pytest.skip("needs credentials or data which is not in the repository")

    papermill = pytest.importorskip("papermill")

    # The notebooks use relative paths, therefore papermill runs them in their own
    # folder. The prepared and the executed copy go into the temporary test folder.
    papermill.execute_notebook(
        input_path=str(_copy_without_skipped_cells(notebook, tmp_path / notebook.name)),
        output_path=str(tmp_path / f"executed_{notebook.name}"),
        cwd=str(notebook.parent),
    )
