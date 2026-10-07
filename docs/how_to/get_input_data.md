# Get the Data for Your Simulations


ETHOS.RESKit provides you with the required input data for your simulation. Datasets that are freely
distributable are downloaded automatically in the examples provided. Please note you are responsible to 
check if the licenses of the data support your usecase.

## Get the inputs a workflow needs


The data collections to download are named by the workflow that reuqires them.
data.paths automatically fetches the data into your local cache. By settinng test=True or False you
can selelct to download the full dataset or a subset of the data.

```python
import reskit as rk
import pandas as pd
from reskit import data

inputs = data.paths("wind_era5_PenaSanchezDunkelWinklerEtAl2025", test=True)
placements = pd.read_csv(data.paths("example_placements")["turbines"])

result = rk.wind.wind_era5_PenaSanchezDunkelWinklerEtAl2025(
    placements=placements,  # prepared as in the wind workflow example
    era5_path=inputs["era5"],
    gwa_100m_path=inputs["gwa_100m"],
    height_scaling_data={50: inputs["gwa_50m"], 200: inputs["gwa_200m"]},
)
```

For further application examples take a look at workflow notebooks of reskit.
```

### Optional command-line access

The CLI is useful for inspecting inputs, planning a download or filling a cache
before running Python:

```bash
reskit-data show wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test
reskit-data fetch wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --plan
reskit-data fetch wind_era5_PenaSanchezDunkelWinklerEtAl2025 --test --paths
```

`show` describes the collection and its named inputs and downloads nothing.
`fetch` transfers data; `--plan` previews the transfer instead of running it.
`--paths` prints one `handle<TAB>absolute path` line per input once the files are



