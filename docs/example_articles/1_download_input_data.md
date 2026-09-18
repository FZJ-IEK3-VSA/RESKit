# Obtain and prepare RESKit input data

For catalogued workflow inputs, start with
[Get input data from the ETHOS.Data catalogue](../how_to/get_input_data.md).
It uses `reskit.data` and `reskit-data` to select, fetch and verify RESKit's inputs.

The examples below cover obtaining upstream weather data and preparing it for
RESKit. Use [development staging](../how_to/get_input_data.md#develop-against-unpublished-data)
when testing a prepared dataset before catalogue acceptance.

## Time resolved Weather data

1. ERA5 Dataset https://doi.org/10.24381/cds.adbb2d47. ERA5 is an abbreviation for ECMWF Reanalysis v5, which stands for the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5.
   1.  [Download the ERA 5 Data](../examples/1_load_input_data/1_1_1_how_to_download_era5_data.ipynb)
   2.  [Calculate the absolute Windspeeds from the ERA5 data](../examples/1_load_input_data/1_1_2_wind_speed_from_vectors_in_era5.ipynb)
2. The MERRA Dataset https://doi.org/10.1175/JCLI-D-11-00015.1. Merra stands for Modern Era Retrospective-Analysis for Research and Applications.
   1.  [Show Structure of MERRA data](../examples/1_load_input_data/1_2_1_reading_merra_weather_data.ipynb)
   2.  [Calculate Windspeeds from MERRA data](../examples/1_load_input_data/1_2_2_vertically_project_wind_speed_from_merra.ipynb)
<!-- 3. ICONLAM # TODO Create Example 
1. COSOMO # Needs check for relevance # TODO Create Example -->

<!-- ## Globally spatially Resolved Irrdatioan and Windspeed Data

1. Global Solar Atlas (GSA) # No Example
2. Global Wind Atlas  (GWA) # No Example -->

# Wind Power Curves 
A wind turbine power curve is a chart that shows how much power a turbine generates at various wind speeds.

1. Power Curves from https://www.thewindpower.net/ can be integrated in RESKit as shown in this [example](../examples/1_load_input_data/1_3_1_process_power_curves_from_thewindpower_net.ipynb)
