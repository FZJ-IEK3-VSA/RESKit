"""Names which the two ERA5 download formats give to the same axes.

The legacy ``netcdf_legacy`` export names the time axis ``time``. The CF compliant
``netcdf`` export names it ``valid_time``. Era5Prepare, Era5Source and Era5ZarrSource all
have to accept both, so the candidate names live here in one place.
"""

# Time axis names which an ERA5 file can use, in order of preference.
ERA5_TIME_NAMES = ("time", "valid_time")
