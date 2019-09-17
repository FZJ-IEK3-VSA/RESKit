from distutils.core import setup

setup(
    name='reskit',
    version='0.1.1',
    author='David Severin Ryberg, Dilara Gulcin Caglayan, Sabrina Schmitt, Roman Kraemer',
    url='http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html',
    packages = ["res"],
    install_requires = [
        "geokit>=1.1.3",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pvlib==0.5.1"
    ]
)