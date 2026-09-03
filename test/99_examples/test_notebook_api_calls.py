"""Check the RESKit calls of the example notebooks without an execution.

Some notebooks need credentials, a large download or a purchased dataset, therefore the
CI never executes them. An old API call in such a notebook stays unnoticed until a user
runs it. These tests parse the code cells and compare every RESKit call with the public
API.
"""

import ast
import inspect
import json
from pathlib import Path

import pytest

import reskit

TEST_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = TEST_DIR.parent.parent / "examples"

NOTEBOOKS = sorted(
    notebook for notebook in EXAMPLES_DIR.glob("**/*.ipynb") if ".ipynb_checkpoints" not in notebook.parts
)

# The names which the notebooks give to the reskit module.
RESKIT_NAMES = {"rk", "reskit"}


def _notebook_id(notebook: Path) -> str:
    """Give the notebook path relative to the examples folder as the test id."""
    return notebook.relative_to(EXAMPLES_DIR).as_posix()


def _code_cells(notebook: Path) -> list:
    """Give the code of every code cell, without the IPython magics."""
    content = json.loads(notebook.read_text(encoding="utf-8"))
    cells = []
    for cell in content["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        cells.append("\n".join(line for line in lines if not line.lstrip().startswith(("%", "!", "?"))))
    return cells


def _dotted_name(node) -> list:
    """Give the parts of a dotted name, for example ["rk", "wind", "onshore_turbine"]."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return []
    parts.append(node.id)
    parts.reverse()
    return parts


def _reskit_object(parts: list):
    """Resolve a dotted RESKit name, or give None if one part is absent."""
    obj = reskit
    for part in parts[1:]:
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_notebook_id)
def test_the_notebook_uses_existing_reskit_attributes(notebook):
    """Every "rk.<name>" of a notebook must exist in the reskit package."""
    for code in _code_cells(notebook):
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, ast.Attribute):
                continue
            parts = _dotted_name(node)
            if not parts or parts[0] not in RESKIT_NAMES:
                continue
            assert _reskit_object(parts) is not None, f"reskit has no {'.'.join(parts[1:])}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_notebook_id)
def test_the_notebook_uses_existing_keyword_arguments(notebook):
    """Every keyword of a RESKit call in a notebook must exist in the signature."""
    for code in _code_cells(notebook):
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, ast.Call) or not node.keywords:
                continue
            parts = _dotted_name(node.func)
            if not parts or parts[0] not in RESKIT_NAMES:
                continue

            function = _reskit_object(parts)
            if not callable(function):
                continue  # an absent name is reported by the attribute test
            try:
                signature = inspect.signature(function)
            except (TypeError, ValueError):
                continue
            if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
                continue

            for keyword in node.keywords:
                if keyword.arg is None:
                    continue  # the call gives a dict with **
                assert keyword.arg in signature.parameters, f"{'.'.join(parts)}() has no argument '{keyword.arg}'"
