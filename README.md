<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster" alt="Forschungszentrum Juelich Logo" width="230px"></a> 

# RESKit - **R**enewable **E**nergy **S**imulation tool**kit** for Python

RESKit aids with the broad-scale simulation of renewable energy systems. Models currently exist for onshore and offshore wind turbines, as well as for solar PV systems. Simulations are performed in the context of singular units, however high computational performance is nevertheless maintained. As a result, this tool allows for the simulation of millions of individual turbines and PV systems in a matter of minutes (on the right hardware). 

**NOTE:** The current state of the model is **highly developmental**, and thus should not be considered to be in a finalized state.

TODO: Register on Zenodo

## Features
  * High performance unit-level wind turbine and PV module simulations
	- Can generate synthetic wind turbine power curves
	- Access to all PV modules in the most recent databases from Sandia and the CEC
  * Configurable to make use climate model datasets
  * Flexible & modular function designs			

## Installation

First, be sure the following modules are installed:
  1. netCDF4
  2. PVLib (version 0.5.1) TODO: check that this is the version!
  3. numpy
  4. pandas
  5. gdal (version 2.4.1)
  6. matplotlib
  7. descartes
  8. <a href="https://github.com/FZJ-IEK3-VSA/geokit">GeoKit</a>
	
NOTE: When using anaconda python (which is highly recommended), requirements 1-7 can often be installed at once using:
```bash
	$ conda install -c conda-forge numpy pandas matplotlib descartes gdal==2.4.1 netCDF4 
	$ conda install -c pvlib pvlib==0.5.1 
``` 
NOTE: Requirement 8 should be installed by cloning the repository and following the associated installation instructions.

Second, clone the RESKit repository to your machine. Then install RESKit via pip as follows
```bash
	pip install -e <path-to-repository>
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

Copyright (c) 2019 David Severin Ryberg (FZJ IEK-3), heidi Heinrichs (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

You should have received a copy of the MIT License along with this program.  
If not, see <https://opensource.org/licenses/MIT>

## About Us 
<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://fz-juelich.de/SharedDocs/Bilder/IEK/IEK-3/Abteilungen2015/VSA_DepartmentPicture_2017.jpg?__blob=normal" alt="Abteilung VSA"></a> 

We are the [Process and Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the Forschungszentrum Jülich. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.


## Acknowledgment

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050 – A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
