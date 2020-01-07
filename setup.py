from setuptools import setup, find_packages

setup(
    name='reskit',
    version='0.1.1',
    author='David Severin Ryberg, Dilara Gulcin Caglayan, Sabrina Schmitt, Roman Kraemer',
    url='https: // github.com/FZJ-IEK3-VSA/reskit',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geokit>=1.1.3",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pvlib==0.5.1"
    ]
)
