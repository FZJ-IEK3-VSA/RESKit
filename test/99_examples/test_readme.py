"""Smoke tests for the python examples of the README.

The tests check that every block compiles and that every documented call matches the
public signature.
"""

import ast
import inspect
import re
from os import path

import pytest

from reskit.workflow_manager import WorkflowManager

README = path.join(path.dirname(__file__), "..", "..", "README.md")

# the callable which each documented call belongs to
DOCUMENTED_CALLS = {
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
    """Return the dotted name of a call, e.g. 'wf.read'."""
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

    assert checked >= len(DOCUMENTED_CALLS), "The README calls which the test knows about were not found."
