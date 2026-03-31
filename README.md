| Version                                                                                                             | Zenodo Release                                                                                              | Docstring Style                                                                  |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [![Conda Version](https://img.shields.io/conda/vn/conda-forge/reskit.svg)](https://anaconda.org/conda-forge/reskit) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17668775.svg)](https://doi.org/10.5281/zenodo.17668775) | ![Numpy docstring Style](https://img.shields.io/badge/%20style-numpy-459db9.svg) |
# RESKit - **R**enewable **E**nergy **S**imulation tool**kit** for Python

<p float="left">
<a href="https://www.fz-juelich.de/en/ice/ice-2"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/JSA-Header.svg?raw=True" alt="Jülich Systems Analysis Logo" width="300px"></a>
</p>

RESKit aids with the broad-scale simulation of renewable energy systems, primarily for the purpose of input generation to Energy System Design Models. Simulation tools currently exist for onshore and offshore wind turbines, as well as for solar photovoltaic (PV) systems and concentrated solar power (CSP), in addition to general weather-data manipulation tools. Simulations are performed in the context of singular units, however high computational performance is nevertheless maintained. As a result, this tool allows for the simulation of millions of individual turbines and PV/CSP systems in a matter of minutes depending on the hardware.

## Features

- High performance unit-level wind turbine, PV module and CSP simulations
- Can generate synthetic wind turbine power curves
- Access to all PV modules in the most recent databases from Sandia and the California Energy Commission (CEC)
- Configurable to make use of different climate model datasets
- Allows correction to real national capacity factor averages
- Flexible & modular function designs

## Installation

#### For Users (Application Only)

1 a) If you do not have an existing conda/mamba environment:
```
conda env create -c conda-forge reskit -n <ENVIRONMENT-NAME>
```

1 b) If you have an existing environment, install RESKit into it:
```
conda install -c conda-forge reskit -n <YOUR-ENVIRONMENT-NAME>
```

2 ) Activate the environment:
```
conda activate <YOUR-ENVIRONMENT-NAME>
```

3 a) Get the RESKit source code (including examples):
```
git clone https://github.com/FZJ-IEK3-VSA/reskit.git
cd reskit
```

3 b) If you do not have Git and just want to check the examples, download and extract the source code with this link:
> https://github.com/FZJ-IEK3-VSA/RESKit/archive/refs/heads/master.zip


#### For Developers
Please follow these steps for an editable installation:

1 ) Clone and checkout dev:

```
git clone https://github.com/FZJ-IEK3-VSA/reskit.git
cd reskit
git checkout dev
```

2 a) RESkit should be installable to a new environment with:

```
conda env create --file requirements-dev.yml
```

2 b) (Alternative) Or into an existing environment with:

```
conda env update --file requirements-dev.yml -n <ENVIRONMENT-NAME>
```

3 ) Install an editable version of reskit (when in the reskit folder) via
```
pip install -e .
```

## ERA5 Zarr

RESKit can read ERA5 directly from regular latitude/longitude Zarr stores while keeping the existing `source_type="ERA5"` workflow API. The current implementation is intended for stores such as the [Earth Data Hub ERA5 single-level dataset](https://earthdatahub.destine.eu/collections/era5/datasets/reanalysis-era5-single-levels):

```python
wf.read(
    variables=["surface_pressure", "surface_air_temperature", "elevated_wind_speed"],
    source_type="ERA5",
    source="https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
    chunks={"time": 48},
    time_slice=slice("2020-01-01", "2020-01-31 23:00:00"),
    set_time_index=True,
)
```

Current limitations:
- The implementation only supports regular `(time|valid_time, latitude, longitude)` Zarr layouts, not flattened `values`-based ERA5 archives.
- If the Zarr store does not ship RESKit's processed `ssrd_t_adj` and `fdir_t_adj` fields, `global_horizontal_irradiance` and `direct_horizontal_irradiance` fall back to raw `ssrd` and `fdir`.


## Citation

If you decide to use RESkit anywhere in a published work related to wind energy, please kindly cite us using the following publications.

When using the ETHOS.RESKit.Wind workflow please cite: 
```bibtex
@article{PenaSanchezDunkelWinklerEtAl2026,
  title = {Towards High Resolution, Validated and Open Global Wind Power Assessments},
  author = {{Pe{\~n}a-S{\'a}nchez}, E. U. and Dunkel, P. and Winkler, C. and Heinrichs, H. and Prinz, F. and Weinand, J. M. and Maier, R. and Dickler, S. and Chen, S. and Gruber, K. and Kl{\"u}tz, T. and Lin{\ss}en, J. and Stolten, D.},
  year = 2026,
  month = jan,
  journal = {Nature Communications},
  volume = {17},
  number = {1},
  pages = {539},
  issn = {2041-1723},
  doi = {10.1038/s41467-026-68337-z},
  url = {http://dx.doi.org/10.1038/s41467-026-68337-z},
}
```

When using anything else:
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

The source code in this repository is licensed under: MIT License Copyright (c) 2019-2025 FZJ-ICE-2

The data files [cf_correction_factors_PSDW2025.tif](reskit/wind/core/data/cf_correction_factors_PSDW2025.tif) and [ws_correction_factors_PSDW2025.yaml](reskit/wind/core/data/ws_correction_factors_PSDW2025.yaml) are licensed under CC-BY-4.0

You should have received a copy of the MIT License along with this program.  
If not, see <https://opensource.org/licenses/MIT>

## About Us 

We are the <a href="https://www.fz-juelich.de/de/ice/ice-2">Institute of Energy and Climate Research - Jülich Systems Analysis (ICE-2)</a> belonging to the <a href="https://www.fz-juelich.de/en">Forschungszentrum Jülich</a>. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Contributions and Support
Every contributions are welcome:
- If you want to report a bug, please open an [Issue](https://github.com/FZJ-IEK3-VSA/RESKit/issues/new). We will then take care of the issue as soon as possible.
- If you want to contribute with additional features or code improvements, open a [Pull request](https://github.com/FZJ-IEK3-VSA/RESKit/pulls).

## Code of Conduct
Please respect our [code of conduct](./docs/CODE_OF_CONDUCT.md).

## Acknowledgement
This work was initially supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/). 

<p float="left">
<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px"></a>
</p>
