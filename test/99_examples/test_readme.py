"""Smoke tests for the python examples of the README.

The tests check that every documented call matches the public signature, and they run the
'download_and_process' example up to the external download boundary. The weather source
preparers are replaced by stubs, so no provider is contacted.
"""

import ast
import inspect
import re
from os import path

import pytest

import reskit as rk
from reskit.util import input_preparation
from reskit.workflow_manager import WorkflowManager

README = path.join(path.dirname(__file__), "..", "..", "README.md")

# the callable which each documented call belongs to
DOCUMENTED_CALLS = {
    "rk.download_and_process": rk.download_and_process,
    "wf.read": WorkflowManager.read,
}


def _python_blocks():
    """Return the python code blocks of the README."""
    with open(README, encoding="utf-8") as fo:
        text = fo.read()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks, "No python code block was found in the README."
    return blocks


def _call_name(node):
    """Return the dotted name of a call, e.g. 'rk.download_and_process'."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    if isinstance(func, ast.Name):
        return func.id
    return None


@pytest.mark.parametrize("index", range(len(_python_blocks())))
def test_readme_block_compiles(index):
    compile(_python_blocks()[index], f"README.md:block{index}", "exec")


def test_readme_calls_match_the_public_signatures():
    checked = 0
    for index, block in enumerate(_python_blocks()):
        for node in ast.walk(ast.parse(block)):
            if not isinstance(node, ast.Call):
                continue
            function = DOCUMENTED_CALLS.get(_call_name(node))
            if function is None:
                continue
            keywords = {keyword.arg: None for keyword in node.keywords if keyword.arg is not None}
            signature = inspect.signature(function)
            unknown = [
                name
                for name in keywords
                if name not in signature.parameters
                and not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
            ]
            assert not unknown, f"README block {index} calls {_call_name(node)} with unknown keywords: {unknown}"
            # bind to prove the documented keywords are accepted
            signature.bind_partial(**keywords)
            checked += 1

    assert checked >= 2, "The README calls which the test knows about were not found."


def test_readme_download_and_process_example_runs(monkeypatch, capsys):
    # replace the preparers, the example must not contact a provider
    recorded = {}

    def _fake_era5(variables, **context):
        recorded["era5"] = dict(variables=list(variables), **context)
        return {"era5_path": path.join(context["output_dir"], "<ZOOM>", "<X-TILE>", "<Y-TILE>")}

    def _fake_gwa4(variables, **context):
        recorded["gwa4"] = list(variables)
        return None

    monkeypatch.setitem(input_preparation._SOURCE_PREPARERS, "ERA5", _fake_era5)
    monkeypatch.setitem(input_preparation._SOURCE_PREPARERS, "GWA4", _fake_gwa4)

    block = next(block for block in _python_blocks() if "download_and_process" in block)
    exec(compile(block, "README.md:download_and_process", "exec"), {})

    # the example prints result["era5_path"], so the key must exist
    assert "weather_data" in capsys.readouterr().out
    # the documented workflow reaches both of its sources
    assert recorded["era5"]["start_date"] == "2000-01-01"
    assert recorded["era5"]["tiling"] is True
    assert recorded["gwa4"]


def test_download_and_process_still_accepts_the_old_keyword(monkeypatch):
    # the README used 'workflow=' before, that spelling must warn but still work
    monkeypatch.setitem(input_preparation._SOURCE_PREPARERS, "ERA5", lambda variables, **context: {"era5_path": "x"})
    monkeypatch.setitem(input_preparation._SOURCE_PREPARERS, "GWA4", lambda variables, **context: None)

    with pytest.warns(DeprecationWarning):
        result = rk.download_and_process(
            workflow="wind_era5_PenaSanchezDunkelWinklerEtAl2025",
            start_date="2000-01-01",
            end_date="2000-12-31",
            boundary_box={"north": 55, "south": 47, "west": 6, "east": 15},
            output_dir="/path/to/your/weather_data",
        )

    assert result == {"era5_path": "x"}


def test_download_and_process_rejects_both_spellings():
    with pytest.raises(ValueError):
        rk.download_and_process(
            workflows="openfield_pv_era5",
            workflow="openfield_pv_era5",
            start_date="2000-01-01",
            end_date="2000-12-31",
            boundary_box={"north": 55, "south": 47, "west": 6, "east": 15},
            output_dir="/tmp",
        )


def test_download_and_process_requires_the_mandatory_arguments():
    with pytest.raises(ValueError):
        rk.download_and_process(workflows="openfield_pv_era5")
