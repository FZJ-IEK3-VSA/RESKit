![image info]()
<p float="left">
<a href="https://www.fz-juelich.de/en/ice/ice-2"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/JSA-Header.svg" alt="Jülich Systems Analysis Logo" width="200px"></a>
</p>

# RESKit - **R**enewable **E**nergy **S**imulation tool**kit** for Python

RESKit aids with the broad-scale simulation of renewable energy systems, primarily for the purpose of input generation to Energy System Design Models. Simulation tools currently exist for onshore and offshore wind turbines, as well as for solar photovoltaic (PV) systems and concentrated spöar power (CSP), in addtion to general weather-data manipulation tools. Simulations are performed in the context of singular units, however high computational performance is nevertheless maintained. As a result, this tool allows for the simulation of millions of individual turbines and PV/CSP systems in a matter of minutes depending on the hardware.

## Features

- High performance unit-level wind turbine, PV module and CSP simulations
- Can generate synthetic wind turbine power curves
- Access to all PV modules in the most recent databases from Sandia and the California Energy Commission (CEC)
- Configurable to make use of different climate model datasets
- Allows correction to real national capacity factor averages
- Flexible & modular function designs

## Installation

Please follow these steps for an editable installation:

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
conda env create --file requirements-dev.yml
```

2. (Alternative) Or into an existing environment with:

```
conda env update --file requirements-dev.yml -n <ENVIRONMENT-NAME>
```

3. Install an editable version of reskit (when in the reskit folder) via
```
pip install -e .
```

## Examples

See the [Examples page](Examples/)

If you intend to use **ETHOS.RESKit.Wind** please follow the following instructions:
1. Download ERA5 and further datasets: 
  1.1. Download ERA5: [How_to_download_ERA5_data_public.ipynb](Examples/How_to_download_ERA5_data_public.ipynb)
  1.2. Process Wind Speeds:  [WindSpeed_from_vectors_public.ipynb](Examples/WindSpeed_from_vectors_public.ipynb)
  1.3. Download ESA Land Cover CCI as tif file: maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf 
  1.4. Download GWAv3: https://globalwindatlas.info/api/gis/global/wind-speed/100 (https://globalwindatlas.info/en/download/gis-files)
1. (optional) If you have purchased PowerCurves from thewindpower.net, please use the following script to process them: [ETHOS.RESKit.Wind_process_power_curves.ipynb](Examples/ETHOS.RESKit.Wind_process_power_curves.ipynb)
2. You can find an example on how to use ETHOS.RESKit.Wind here: [ETHOS.RESKit.Wind.ipynb](Examples/ETHOS.RESKit.Wind.ipynb)


## Citation

If you decide to use RESkit anywhere in a published work related to wind energy, please kindly cite us using the following

When using ETHOS.RESKit.Wind workflow please use: 
```bibtex
@article{PenaSanchezDunkelWinklerEtAl2025,
      title={Towards high resolution, validated and open global wind power assessments}, 
      author={Edgar Ubaldo Peña-Sánchez and Philipp Dunkel and Christoph Winkler and Heidi Heinrichs and Florian Prinz and Jann Weinand and Rachel Maier and Sebastian Dickler and Shuying Chen and Katharina Gruber and Theresa Klütz and Jochen Linßen and Detlef Stolten},
      year={2025},
      eprint={2501.07937},
      archivePrefix={arXiv},
      primaryClass={physics.soc-ph},
      url={https://arxiv.org/abs/2501.07937}, 
}
```

When using other anything else:
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

Copyright (c) 2021-2025 FZJ-ICE-2

You should have received a copy of the MIT License along with this program.  
If not, see <https://opensource.org/licenses/MIT>

## About Us 

We are the <a href="https://www.fz-juelich.de/de/ice/ice-2">Institute of Energy and Climate Research - Jülich Systems Analysis (ICE-2)</a> belonging to the <a href="https://www.fz-juelich.de/en">Forschungszentrum Jülich</a>. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Contributions and Support
Every contributions are welcome:
- If you want to report a bug, please open an [Issue](https://github.com/FZJ-IEK3-VSA/RESKit/issues/new). We will then take care of the issue as soon as possible.
- If you want to contribute with additional features or code improvements, open a [Pull request](https://github.com/FZJ-IEK3-VSA/RESKit/pulls).

## Code of Conduct
Please respect our [code of conduct](CODE_OF_CONDUCT.md).

## Acknowledgement
This work was initially supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/). 

<p float="left">
<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px"></a>
</p>
