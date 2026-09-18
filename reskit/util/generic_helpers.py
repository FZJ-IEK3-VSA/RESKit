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

    lengths = [
        1 if arr.ndim == 0 else len(arr)
        for arr in arrays
    ]
    n = max(lengths)

    aligned = []
    for arr in arrays:
        if arr.ndim == 0:
            arr = np.full(n, arr.item())
        elif len(arr) == 1 and n > 1:
            arr = np.full(n, arr[0])
        elif len(arr) != n:
            raise ValueError(
                f"All non-scalar inputs must have length 1 or {n}, "
                f"here found length {len(arr)}."
            )

        aligned.append(arr)

    return (*aligned, scalar)


