import inspect
import numpy as np
import warnings


def _align_inputs(*args):
    """
    Align arbitrary scalar/array-like inputs to the same 1D length.

    Scalars and length-1 inputs are repeated to match the longest input.
    All other iterable inputs must already have the maximum length.

    Returns
    -------
    tuple
        Aligned inputs as np.ndarray objects, followed by scalar flag.
    """
    arrays = [np.asarray(arg) for arg in args]
    scalar = all(arr.ndim == 0 for arr in arrays)

    lengths = [1 if arr.ndim == 0 else len(arr) for arr in arrays]
    n = max(lengths)

    aligned = []
    for arr in arrays:
        if arr.ndim == 0:
            arr = np.full(n, arr.item())
        elif len(arr) == 1 and n > 1:
            arr = np.full(n, arr[0])
        elif len(arr) != n:
            raise ValueError(f"All non-scalar inputs must have length 1 or {n}, here found length {len(arr)}.")

        aligned.append(arr)

    return (*aligned, scalar)


def _check_kwargs(func, kwargs, raise_error=True, raise_warnings=True):
    """
    Check kwargs against the signature of a callable and returns
    missing and obsolete arguments separately as a dict, unless
    error shall be raised on missing mandatory arguments.

    func : Callable
        The function wgainst which the kwargs shall be checked
    kwargs : dict
        The kwargs and values that shall be checked against the
        func signature.
    raise_error : bool, optional
        Will raise an error on missing mandatory arguments, by
        default True.
    raise_warnings : bool, optional
        Will raise warnings on missing optional arguments and
        extra arguments, by default True.

    Returns
    -------
    dict
        {
            "missing_mandatory": {arg: type},
            "missing_optional": {arg: default},
            "extra": {arg: value},
        }
    """
    if not callable(func):
        raise TypeError(f"func must be callable, here: {type(func)}")
    if not isinstance(kwargs, dict):
        raise TypeError(f"kwargs must be dict, here: {type(kwargs)}")
    if not isinstance(raise_error, bool):
        raise TypeError(f"raise_error must be bool, here: {type(raise_error)}")
    if not isinstance(raise_warnings, bool):
        raise TypeError(f"raise_warnings must be bool, here: {type(raise_warnings)}")

    signature = inspect.signature(func)
    params = signature.parameters

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    missing_mandatory = {}
    missing_optional = {}

    for name, param in params.items():
        # *args and **kwargs are never mandatory
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name not in kwargs:
            if param.default is inspect.Parameter.empty:
                arg_type = None if param.annotation is inspect.Parameter.empty else param.annotation
                missing_mandatory[name] = arg_type
            else:
                missing_optional[name] = param.default

    if accepts_kwargs:
        extra = {}
    else:
        extra = {name: value for name, value in kwargs.items() if name not in params}

    if raise_error and missing_mandatory:

        def _format_annotation(annotation):
            if annotation is None:
                return "(type not specified)"
            if isinstance(annotation, type):
                return annotation.__name__
            return str(annotation)

        missing_str = ", ".join(
            f"'{name}' : {_format_annotation(arg_type)}" for name, arg_type in missing_mandatory.items()
        )
        raise TypeError(f"Missing mandatory argument(s) for '{getattr(func, '__name__', str(func))}': {missing_str}")

    if raise_warnings and missing_optional:
        missing_optional_str = ", ".join(f"'{name}' : {default!r}" for name, default in missing_optional.items())
        warnings.warn(
            f"The following optional argument(s) for "
            f"'{getattr(func, '__name__', str(func))}' have not been provided, "
            f"default value(s) will be used as: {missing_optional_str}",
            UserWarning,
        )

    if raise_warnings and extra:
        extra_str = ", ".join(f"'{name}' : {value!r}" for name, value in extra.items())
        warnings.warn(
            f"Extra keyword argument(s) passed to "
            f"'{getattr(func, '__name__', str(func))}' (cannot not be applied): {extra_str}",
            UserWarning,
        )

    return {
        "missing_mandatory": missing_mandatory,
        "missing_optional": missing_optional,
        "extra": extra,
    }
