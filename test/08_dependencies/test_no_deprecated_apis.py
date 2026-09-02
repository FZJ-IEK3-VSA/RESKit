"""Guards against third-party APIs which the upstream project announced for removal.

The runtime gate in ``pyproject.toml`` turns such a deprecation warning into an
error. That gate only sees a line which a test executes. The tests here read the
RESKit source instead, so they also cover the lines which no test executes.

Add a guard here when an upstream project announces a removal which RESKit code
touches. Record the upstream version in the docstring of the guard.
"""

import ast
from pathlib import Path

import reskit

RESKIT_ROOT = Path(reskit.__file__).parent


def _source_files():
    """All python files of the RESKit package."""
    return sorted(RESKIT_ROOT.rglob("*.py"))


def _calls_with_keyword(path, keyword):
    """Report every call in ``path`` which gives ``keyword`` as a keyword argument.

    The check reads the syntax tree, not the text. A comment or a string with the
    same name does not count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for argument in node.keywords
        if argument.arg == keyword
    )


def test_no_call_uses_the_pvlib_apparent_azimuth_keyword():
    """The pvlib project renamed ``apparent_azimuth`` to ``solar_azimuth`` in 0.13.1.

    pvlib removes the old name soon. The parameter keeps its position in both
    names, so RESKit gives the argument by position. A position works on every
    pvlib version which ``requirements.yml`` supports.
    """
    offenders = [
        f"{path.relative_to(RESKIT_ROOT.parent)}:{line}"
        for path in _source_files()
        for line in _calls_with_keyword(path, "apparent_azimuth")
    ]
    assert offenders == [], (
        "These calls use the pvlib keyword 'apparent_azimuth', which pvlib removes soon. "
        "Give the argument by position instead: " + ", ".join(offenders)
    )
