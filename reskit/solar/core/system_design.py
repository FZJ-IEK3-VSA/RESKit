import ast
import operator
import warnings
from os.path import isfile

import geokit as gk
import numpy as np

from ...util import ResError

# The names which a tilt convention expression is permitted to use
_TILT_EXPRESSION_NAMES = {
    "abs": np.abs,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "cos": np.cos,
    "degrees": np.degrees,
    "exp": np.exp,
    "log": np.log,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "pi": np.pi,
    "radians": np.radians,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tan": np.tan,
}

# The operators which a tilt convention expression is permitted to use
_TILT_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_TILT_UNARY_OPERATORS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_tilt_expression(expression, latitude):
    """Evaluate a tilt convention expression without ``eval``.

    ``eval`` gives the Python built-ins to the expression. An untrusted convention string
    can therefore run arbitrary code. This function accepts only numbers, the variable
    'latitude', the arithmetic operators, and the functions of ``_TILT_EXPRESSION_NAMES``.

    Parameters
    ----------
    expression : str
        The expression to evaluate, e.g. "latitude*0.76".

    latitude : numpy.ndarray
        The latitude values which the name 'latitude' refers to.

    Returns
    -------
    numpy.ndarray or float
        The result of the expression.

    Raises
    ------
    ResError
        If the expression is not valid Python, or if it uses a name, a constant, or an
        operator which is not permitted.
    """
    names = dict(_TILT_EXPRESSION_NAMES)
    names["latitude"] = latitude

    def _evaluate(node):
        if isinstance(node, ast.Expression):
            return _evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ResError(f"Tilt convention expression must not contain the constant: {node.value!r}")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ResError(f"Tilt convention expression must not use the name: '{node.id}'")
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _TILT_UNARY_OPERATORS:
            return _TILT_UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _TILT_BINARY_OPERATORS:
            return _TILT_BINARY_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _TILT_EXPRESSION_NAMES:
                raise ResError("Tilt convention expression must only call the permitted functions.")
            if node.keywords:
                raise ResError("Tilt convention expression must not use keyword arguments.")
            return _TILT_EXPRESSION_NAMES[node.func.id](*[_evaluate(arg) for arg in node.args])
        raise ResError(f"Tilt convention expression contains an element which is not permitted: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ResError(f"Tilt convention is neither a known convention, nor a file, nor an expression: {expression}")

    return _evaluate(tree)


def location_to_tilt(locs, convention="Ryberg2020", **kwargs):
    """
    Simple system tilt estimator based off latitude and longitude coordinates


    Parameters
    ----------
    locs : geokit.LocationSet or iterable of (lon,lat) pairs
           The locations at which to estimate system tilt angle

    convention : str or callable, optional
                 The calculation method used to suggest system tilts
                 Options are:
                     * "Ryberg2020"
                     * A callable which takes the latitude array and returns the tilts
                     - Ex. lambda latitude: latitude * 0.76
                     * A path to a raster file
                     * A restricted arithmetic expression string (deprecated)
                     - Can use the variable 'latitude' and the functions of
                       `_TILT_EXPRESSION_NAMES`
                     - Ex. "latitude*0.76"

    kwargs: Optional keyword arguments to use in geokit.raster.interpolateValues(...).
            Only applies when `convention` is a path to a raster file


    Returns
    -------
    np.ndarray
        Suggested tilt angle at each of the provided `locs`.
        Has the same length as the number of `locs`.

    Notes
    -----
    "Ryberg2020"
        When `convention` equals "Ryberg2020", the following equation is followed:

        .. math:: 42.327719357601396 * arctan( 1.5 * abs(latitude) )

    .. [1] TODO: Cite future Ryberg2020 publication

    """
    locs = gk.LocationSet(locs)

    if callable(convention):
        tilt = convention(locs.lats)

    elif not isinstance(convention, str):
        raise ResError(f"Tilt convention must be a string or a callable, but is: {type(convention)}")

    elif convention == "Ryberg2020":
        tilt = 42.327719357601396 * np.arctan(1.5 * np.radians(np.abs(locs.lats)))

    elif isfile(convention):
        tilt = gk.raster.interpolateValues(convention, locs, **kwargs)

    else:
        warnings.warn(
            "String tilt convention expressions are deprecated and they will be removed. "
            "Pass a callable instead, e.g. convention=lambda latitude: latitude * 0.76.",
            DeprecationWarning,
            stacklevel=2,
        )
        tilt = _evaluate_tilt_expression(convention, locs.lats)

    return tilt
