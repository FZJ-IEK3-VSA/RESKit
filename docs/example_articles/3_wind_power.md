# Wind Power Simulation

RESKit provides a feature-rich wind power simulation, as demonstrated by the following examples:

1. The [first example](../../examples/3_wind/3_1_wind_workflows_overview.ipynb) provides an overview of how to apply the available wind power worklows.
2. The next example deals with wind power curves.
   1. [It shows](../../examples/3_wind/3_2_1_wind_load_power_curve.ipynb) how to define, load, and inspect a power curve.
   2. Then, [it presents](../../examples/3_wind/3_2_1_wind_load_power_curve.ipynb) how to create a power curve from srcatch and how to load a real turbine's power curve.
3. The [third example](../../examples/3_wind/3_3_wind_convolute_a_power_curve.ipynb) shows how to convolute the power curve to account for stochastic effects.
4. The [fourth example](../../examples/3_wind/3_4_wind_design_onshore_turbine.ipynb) shows how RESKit can suggest wind turbine designs for specific operation conditions.
5. The [fifth example](../../examples/3_wind/3_5_wind_estimate_turbine_capital_cost.ipynb) demonstrates how  to estimate the capital cost of a wind turbine using RESKit.
6. The sixth example shows how to conduct actual simulations of a wind power plant.
   1. First, a [basic simulation](../../examples/3_wind/3_6_1_wind_basic_turbine_simulation.ipynb) is shown.
   2. Then, a [more advanced example](../../examples/3_wind/3_6_2_wind_turbine_simulation_workflow.ipynb) is shown, which accounts for more realistic effects.
   3. Lastly, [it is demonstrated](../../examples/3_wind/3_6_3_wind_automated_turbine_simulation_workflow.ipynb) how to use one of the predefined workflows provided by RESKit.
      1. To apply this workflow outside of the test example, you need the full ESA Land Cover Dataset. It can be downloaded here: https://cds.climate.copernicus.eu/datasets/satellite-land-cover?tab=download.
7. [The final example](../../examples/3_wind/3_7_example_ethos_reskit_wind_workflow.ipynb) shows how to apply the ETHOS.RESKit.Wind workflow.
   1. This requires the ERA5 dataset, for which a download instructions can be found [here](../../examples/1_load_input_data/1_1_1_how_to_download_era5_data.ipynb).
   2. In order to use the workflow, the ERA5 dataset must be preprocessed to contain the absolute wind speeds, as shown [here](../../examples/1_load_input_data/1_1_2_wind_speed_from_vectors_in_era5.ipynb).
   3. To increase spatial resolution of the simulation and allow extrapolation of wind speeds to turbine hub height, the GWAv4 (Global Wind Atlas) is used. This example provides a small test data sample for demonstration purposes. For applications outside the test example, however, the full GWAv4 dataset is required, which can be downloaded here: https://globalwindatlas.info/en/download/gis-files.
   4. (Optional) If you have purchased Power Curves from thewindpower.net, please use the following script to process them: [Process Power Curves](../../examples/1_load_input_data/1_3_1_process_power_curves_from_thewindpower_net.ipynb)

Paths to a custom turbine library and baseline turbine definitions can be set in [default_paths.yaml](../../reskit/default_paths.yaml).
