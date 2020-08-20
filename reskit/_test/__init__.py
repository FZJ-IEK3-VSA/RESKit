# Add useful paths for testing and stuff
from collections import OrderedDict
from glob import glob
from os.path import join, dirname, basename

TEST_DATA = OrderedDict()

for f in glob(join(dirname(__file__), "data", "*")):
    TEST_DATA[basename(f)] = f
