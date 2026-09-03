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
