| Version                                                                                                             | Zenodo Release                                                                                              | Docstring Style                                                                  | Coverage                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Conda Version](https://img.shields.io/conda/vn/conda-forge/reskit.svg)](https://anaconda.org/conda-forge/reskit) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17668775.svg)](https://doi.org/10.5281/zenodo.17668775) | ![Numpy docstring Style](https://img.shields.io/badge/%20style-numpy-459db9.svg) | [![codecov](https://codecov.io/gh/FZJ-IEK3-VSA/RESKit/branch/master/graph/badge.svg)](https://codecov.io/gh/FZJ-IEK3-VSA/RESKit) |
# RESKit - **R**enewable **E**nergy **S**imulation tool**kit** for Python

<a href="https://www.fz-juelich.de/en/ice/ice-2">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/FZJ-IEK3-VSA/README_assets/blob/v.1.0.0/ICE2_Logos/JSA-Header-dark.svg?raw=True">
    <img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/v.1.0.0/ICE2_Logos/JSA-Header.svg?raw=True" alt="Logo für Forschungszentrum Juelich - Juelich System Analysis" width="300px">
  </picture>
</a>

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
conda env create --file requirements.yml
```

2 b) (Alternative) Or into an existing environment with:

```
conda env update --file requirements.yml -n <ENVIRONMENT-NAME>
```

3 ) Install an editable version of reskit (when in the reskit folder) via
```
pip install -e .
```

## ERA5 Zarr

RESKit can read ERA5 directly from regular latitude/longitude Zarr stores while keeping the existing `source_type="ERA5"` workflow API. The current implementation is intended for stores such as the [Earth Data Hub ERA5 single-level dataset](https://earthdatahub.destine.eu/collections/era5/datasets/reanalysis-era5-single-levels):

Earth Data Hub requires authentication. Follow the credential instructions on the linked dataset page and save the generated credentials as `~/.netrc` (not in a directory on `PATH`). On shared systems, restrict access with `chmod 600 ~/.netrc`. The HTTPS backend will use those credentials automatically.

```python
wf.read(
    variables=["surface_pressure", "surface_air_temperature", "elevated_wind_speed"],
    source_type="ERA5",
    source="https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
    chunks={"valid_time": 48},
    time_slice=slice("2020-01-01", "2020-01-31 23:00:00"),
    set_time_index=True,
)
```

Current limitations:
- The implementation only supports regular `(time|valid_time, latitude, longitude)` Zarr layouts, not flattened `values`-based ERA5 archives.
- If the Zarr store does not ship RESKit's processed `ssrd_t_adj` and `fdir_t_adj` fields,  `global_horizontal_irradiance` and `direct_horizontal_irradiance` fall back to processing the raw `ssrd` and `fdir` on the fly.


## Preparing Weather Data

RESKit workflows are driven by gridded weather data. RESKit ships a single
high-level helper, `rk.download_and_process`, that downloads exactly the variables a
given workflow needs from the relevant data provider, preprocesses them (e.g. wind
speed from u/v components, solar unit and time-shift corrections), and optionally
tiles them into the `<zoom>/<x>/<y>/<year>/` directory structure expected by the
weather sources.

ERA5 reanalysis from the Copernicus Climate Data Store (CDS) is currently the
supported source; additional weather data sources are planned.

```python
import reskit as rk

result = rk.download_and_process(
    workflow="wind_era5_PenaSanchezDunkelWinklerEtAl2025",
    start_date="2000-01-01",
    end_date="2000-12-31",
    boundary_box={"north": 55, "south": 47, "west": 6, "east": 15},  # Germany
    output_dir="/path/to/your/weather_data",
    tiling=True,
)
print(result["era5_path"])
```

To prepare data for several workflows in a single call, pass a list of workflow names
as `workflow`; the union of their variable requirements is downloaded and processed at
once, e.g. `workflow=["openfield_pv_era5", "CSP_PTR_ERA5"]`. A CDS account with a
configured `~/.cdsapirc` API key is required for ERA5
(see https://cds.climate.copernicus.eu/how-to-api).

Note that some workflows also rely on data whose automated download is not yet
implemented — solar/CSP workflows on Global Solar Atlas rasters and wind workflows
on Global Wind Atlas rasters. `download_and_process` prints a notice for these and
you must supply the rasters manually.

End-to-end examples live in [examples/1_load_input_data/](examples/1_load_input_data/):
- [1_1_3_prepare_era5_for_wind_workflow.ipynb](examples/1_load_input_data/1_1_3_prepare_era5_for_wind_workflow.ipynb)
- [1_1_4_prepare_era5_for_solar_workflow.ipynb](examples/1_load_input_data/1_1_4_prepare_era5_for_solar_workflow.ipynb)

For full manual control over the raw ERA5/CDS download (variables, area, and
timeframe), see the lower-level example notebook
[1_1_1_how_to_download_era5_data.ipynb](examples/1_load_input_data/1_1_1_how_to_download_era5_data.ipynb).

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

## Contributions and Support
Every contributions are welcome:
- If you want to report a bug, please open an [Issue](https://github.com/FZJ-IEK3-VSA/RESKit/issues/new). We will then take care of the issue as soon as possible.
- If you want to contribute with additional features or code improvements, open a [Pull request](https://github.com/FZJ-IEK3-VSA/RESKit/pulls).

## About Us 

We are the <a href="https://www.fz-juelich.de/en/ice/ice-2">Institute of Climate and Energy Systems – Jülich Systems Analysis (ICE-2)</a> at the <a href="https://www.fz-juelich.de/en"> Forschungszentrum Jülich</a>.
Our work focuses on independent, interdisciplinary research in energy, bioeconomy, infrastructure, and sustainability. We support a just, greenhouse gas–neutral transformation through open models and policy-relevant science.


## Code of Conduct
Please respect our [code of conduct](https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/CODE_CONDUCT.md).



## Acknowledgement
This work was initially supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/). 

<a href="https://www.helmholtz.de/en/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-White-RGB.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-Dark-Blue-RGB.svg">
    <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-Dark-Blue-RGB.svg" alt="Helmholtz Logo" width="200px" style="float:left">
  </picture>
</a>