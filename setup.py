from setuptools import setup, find_packages

setup(
    name='reskit',
    version='0.2.0',
    author='David Severin Ryberg, Dilara Gulcin Caglayan, Sabrina Schmitt, Roman Kraemer',
    url='https: // github.com/FZJ-IEK3-VSA/reskit',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geokit>=1.2.4",
        "numpy",
        "pandas",
        "scipy",
        # "matplotlib",
        "pvlib>=0.7.2",
        "netCDF4>=1.5.3",
        "xarray",
        "gdal==3.4.2"
    ]
)
