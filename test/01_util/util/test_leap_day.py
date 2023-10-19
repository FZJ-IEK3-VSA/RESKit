import numpy as np
import pandas as pd

from reskit.util.leap_day import remove_leap_day


def test_remove_leap_day():
    array_8784 = np.arange(8784)
    array_8760 = np.arange(8760)

    # Test ndarray inputs
    assert remove_leap_day(array_8784).shape == (8760, )
    assert remove_leap_day(array_8760).shape == (8760, )

    # Test list input
    assert remove_leap_day(array_8784.tolist()).shape == (8760, )
    assert remove_leap_day(array_8760.tolist()).shape == (8760, )

    # Test pandas Series
    series_8784 = pd.Series(array_8784, index=pd.date_range(
        "01-01-2000 00:00:00", "31-12-2000 23:00:00", freq="H"))

    fixed_array = remove_leap_day(series_8784)
    assert fixed_array.shape == (8760, )

    assert fixed_array["02-28-2000 00:00:00"] == 1392
    assert fixed_array["03-01-2000 00:00:00"] == 1440

    try:
        fixed_array["02-29-2000 00:00:00"]
        assert False
    except KeyError:
        assert True
    else:
        assert False
