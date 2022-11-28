<a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="https://raw.githubusercontent.com/OfficialCodexplosive/README_Assets/862a93188b61ab4dd0eebde3ab5daad636e129d5/FJZ_IEK-3_logo.svg" alt="FZJ Logo" width="300px"></a>

# RESKit - **R**enewable **E**nergy **S**imulation tool**kit** for Python

RESKit aids with the broad-scale simulation of renewable energy systems, primarily for the purpose of input generation to Energy System Design Models. Simulation tools currently exist for onshore and offshore wind turbines, as well as for solar PV systems, in addtion to general weather-data manipulation tools. Simulations are performed in the context of singular units, however high computational performance is nevertheless maintained. As a result, this tool allows for the simulation of millions of individual turbines and PV systems in a matter of minutes (on the right hardware).

## Features

- High performance unit-level wind turbine and PV module simulations
  - Can generate synthetic wind turbine power curves
  - Access to all PV modules in the most recent databases from Sandia and the CEC
- Configurable to make use climate model datasets
- Flexible & modular function designs

## Installation

The primary dependancies of RESKit are:

1. netCDF4>=1.5.3
2. xarray
3. PVLib>=0.7.2
4. gdal>2.0.0,<3.0.0
5. <a href="https://github.com/FZJ-IEK3-VSA/geokit">GeoKit</a> >= 1.2.4

If you can install these modules on you own, then the reskit module should be easily installable with:

```
pip install git+https://github.com/FZJ-IEK3-VSA/reskit.git#egg=reskit
```

If, on the otherhand, you prefer an automated installation using Anaconda, then you should be able to follow these steps:

1. First clone a local copy of the repository to your computer, and move into the created directory

```
git clone https://github.com/FZJ-IEK3-VSA/reskit.git
cd reskit
```

1. (Alternative) If you want to use the 'dev' branch (or another branch) then use:

```
git checkout dev
```

2. RESkit should be installable to a new environment with:

```
conda env create --file requirements.yml
```

2. (Alternative) Or into an existing environment with:

```
conda env update --file requirements.yml -n <ENVIRONMENT-NAME>
```

2. (Alternative) If you want to install RESKit in editable mode, and also with jupyter notebook and with testing functionalities use:

```
conda env create --file requirements-dev.yml
```

## Examples

See the [Examples page](Examples/)

## Docker

We are looking into making RESKit accessible in a docker container. Check back later for more info!

## Citation

If you decide to use RES anywhere in a published work related to wind energy, please kindly cite us using the following

```bibtex
@article{RybergWind2019,
  author = {Ryberg, David Severin and Caglayan, Dilara Gulcin and Schmitt, Sabrina and Lin{\ss}en, Jochen and Stolten, Detlef and Robinius, Martin},
  doi = {10.1016/j.energy.2019.06.052},
  issn = {03605442},
  journal = {Energy},
  month = {sep},
  pages = {1222--1238},
  title = {{The future of European onshore wind energy potential: Detailed distribution and simulation of advanced turbine designs}},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0360544219311818},
  volume = {182},
  year = {2019}
}

```

## License

MIT License

Copyright (c) 2019-2022 David Severin Ryberg (FZJ IEK-3), Heidi Heinrichs (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

You should have received a copy of the MIT License along with this program.  
If not, see <https://opensource.org/licenses/MIT>

## About Us
<p align="center"><a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="https://github.com/OfficialCodexplosive/README_Assets/blob/master/iek3-wide.png?raw=true" alt="Institut TSA"></a></p>
We are the <a href="https://www.fz-juelich.de/en/iek/iek-3">Institute of Energy and Climate Research - Techno-economic Systems Analysis (IEK-3)</a> belonging to the <a href="https://www.fz-juelich.de/en">Forschungszentrum Jülich</a>. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

We are the [Process and Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Techno-economic Systems Analysis (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the Forschungszentrum Jülich. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Acknowledgment

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050 – A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
