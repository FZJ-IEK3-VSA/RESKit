import res.weather.NCSource
from .MerraSource import MerraSource
from .CordexSource import CordexSource
from .SevCosmoREA6Source import SevCosmoREA6Source


def loadWeatherSource(path, which="MerraSource", **kwargs):
	if which=="MerraSource":
		return MerraSource(path, **kwargs)
	elif which=="CordexSource":
		return CordexSource(path, **kwargs)
	elif which=="SevCosmoREA6Source":
		return SevCosmoREA6Source(path, **kwargs)
	else:
		raise RuntimeError("Source type not known")