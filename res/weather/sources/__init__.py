import res.weather.NCSource
from .MerraSource import MerraSource
from .TrySource import TrySource
from .CordexSource import CordexSource
from .SevCosmoREA6Source import SevCosmoREA6Source


def loadWeatherSource(path, which="MerraSource", **kwargs):
	which = which.lower()
	if which=="merrasource" or which=="merra":
		return MerraSource(path, **kwargs)
	elif which=="trysource" or which=="try":
		return TrySource(path, **kwargs)
	elif which=="cordexsource" or which=="cordex":
		return CordexSource(path, **kwargs)
	elif which=="sevcosmorea6source" or which=="sevcosmorea6":
		return SevCosmoREA6Source(path, **kwargs)
	else:
		raise RuntimeError("Source type not known")