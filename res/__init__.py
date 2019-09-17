from res import util
from res import weather
from res import economic

# Add useful paths for testing and stuff
from collections import OrderedDict as _OrderedDict
from glob import glob as _glob
from os.path import join as _join, dirname as _dirname, basename as _basename

_TEST_DATA_ = _OrderedDict()

for f in _glob(_join(_dirname(__file__), "test", "data", "*")):
    _TEST_DATA_[_basename(f)] = f
