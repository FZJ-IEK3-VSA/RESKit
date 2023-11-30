from setuptools import setup, find_packages

setup(
    name="reskit",
    version="0.2.0",
    author="David Severin Ryberg, Dilara Gulcin Caglayan, Sabrina Schmitt, Roman Kraemer",
    url="https: // github.com/FZJ-IEK3-VSA/reskit",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geokit>=1.4.0",
        "numpy",
        "numba",
        "pandas<2.0.0",
        "scipy",
        # "matplotlib",
        "gdal==3.4.*",
        "pvlib==0.9.0",
        "netCDF4>=1.5.3",
        "xarray",
    ],
)
