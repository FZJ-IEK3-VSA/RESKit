import yaml
from os.path import join, dirname

DEFAULT_PATHS = {}
with open(join(dirname(__file__), "default_paths.yaml")) as f:
    DEFAULT_PATHS = yaml.safe_load(f)
pass
