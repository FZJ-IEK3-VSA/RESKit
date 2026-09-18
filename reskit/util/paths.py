"""Path arguments which arrive as a string or as a ``pathlib.Path``.

RESKit tells a file path apart from a plain value by its type: a raster path and a
long-run-average number reach the same argument, and ``isinstance(x, str)`` decides
which one was meant. ``reskit.data`` hands back ``pathlib.Path`` objects, which
would have fallen through such a check and been read as a value rather than as a
file. These helpers make the check accept both spellings.
"""

import os


def is_path_like(value) -> bool:
    """Whether ``value`` names a file, as a string or as an ``os.PathLike``."""
    return isinstance(value, (str, os.PathLike))


def as_path_string(value) -> str:
    """The string form of a path argument, whichever spelling it arrived in."""
    return os.fspath(value)
