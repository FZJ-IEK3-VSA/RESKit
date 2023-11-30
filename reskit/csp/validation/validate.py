#%%
import os
import reskit as rk
import pandas as pd
import xarray as xr
import numpy as np
from smopy import deg2num
from glob import glob
import matplotlib.pyplot as plt
from scipy import stats

# import fathon
# from fathon import fathonUtils as fu


validation_output_folder = (
    "/storage/internal/data/d-franzmann/03_Diss/01_CSP/results/02_RESKit/02_validation/"
)

# datasetname = 'Dataset_SolarSalt_2030_validation3'
datasetname = "Dataset_Heliosol_2030_validation3"


def main(datasetname, validation_output_folder):
    if "SolarSalt" in datasetname:
        htf_greenius = "Solar_Salt"
    elif "Heliosol" in datasetname:
        htf_greenius = "Heliosol"
    else:
        raise ValueError

    os.makedirs(validation_output_folder, exist_ok=True)

    path_locations = os.path.join(
        os.path.dirname(__file__), "Validation_Locations.xlsx"
    )

    #%% Make a placements dataframe
    placements = pd.read_excel(path_locations)
    placements["area"] = 1144000  # 4576000 #m^2

    #%% Find the ERA 5 tile for each placement
    # set up dummy columns for x and y tiles
    placements["tilex"], placements["tiley"] = deg2num(
        placements["lat"].values, placements["lon"].values, zoom=4
    )

    era5_path = r"/storage/internal/data/gears/weather/ERA5/processed/4/{x}/{y}/2015"

    #%% Run Reskit and save files
    results_list = []
    for i in range(len(placements)):
        placement = placements.iloc[[i]]
        print(
            "Calculating Placement {i} of {max}: {name}".format(
                i=i + 1, max=str(placements.shape[0]), name=str(placement["Name"])
            )
        )
        # get tile numbers
        x = placement["tilex"].iloc[0]
        y = placement["tiley"].iloc[0]

        # run reskit
        out = rk.csp.CSP_PTR_ERA5(
            placements=placement,
            era5_path=era5_path.format(x=x, y=y),
            global_solar_atlas_dni_path="default_cluster",
            datasets=datasetname,
            verbose=True,
            JITaccelerate=False,
            debug_vars=True,
            _validation=True,
            return_self=False,
        )
        out = out.assign_coords(location=out.location + i)
        results_list.append(out)

        print("Reskit done!")

    out_reskit = xr.concat(results_list, dim="location")
    out_reskit.drop("tile x / y").to_netcdf(
        os.path.join(validation_output_folder, f"sim_results_reskit_{htf_greenius}.nc4")
    )
    # %% Load Greenius Data

    path_greenius_glob = os.path.join(
        os.path.dirname(__file__), "greenuis_reference_data", f"*{htf_greenius}*.xlsx"
    )
    paths_greenius = sorted(list(glob(path_greenius_glob)))
    paths_greenius = paths_greenius
    assert len(paths_greenius) == len(placements)

    greenius_ref_dict = {}
    print("loading greenius results")
    for i, path_greenius in enumerate(paths_greenius):
        print(f"{i} of {len(paths_greenius)}")
        df_data_greenius = pd.read_excel(path_greenius, skiprows=range(1, 4),)
        df_data_greenius["Q losses"] = (
            df_data_greenius["Q Heat"]
            + df_data_greenius["Q Vessel"]
            + df_data_greenius["Q Pipe"]
        )
        greenius_ref_dict[i] = df_data_greenius

    #%% Full metric report

    vars_reskit = "HeattoHTF_W|HTF_mean_temperature_C|HeattoPlant_W|Heat_Losses_W|P_heating_W|Parasitics_solarfield_W_el".split(
        "|"
    )
    vars_greenius = "Q abs|T HTFmean|Q out|Q losses|QFP Aux|W parField".split("|")
    metrics = [
        RMSE,
        absError,
        relabsError,
        meanDev,
        relmeanDev,
        stdDifference,
        corrCoefficient,
        Nash_Sutcliffe,
    ]  # , DCCA]
    fs_greenius = [1e-6, 1, 1e-6, 1e-6, 1e-6, 1e-6]

    full_metric_report_list = []
    for i_placement in range(len(placements)):
        location = placements.loc[i_placement, "Name"]
        for var_reskit, var_greenius, f_greenius in zip(
            vars_reskit, vars_greenius, fs_greenius
        ):
            data_reskit = out_reskit.sel(location=i_placement)[var_reskit].values
            data_reskit = data_reskit * f_greenius
            data_greenius = greenius_ref_dict[i_placement][var_greenius].values

            columns = ["region", "variable"]
            values = [location, var_reskit.replace("W", "MW")]
            for metric in metrics:
                columns.append(metric.__name__)
                values.append(metric(data_reskit, data_greenius))
            df_temp = pd.DataFrame([values], columns=columns, index=[i_placement])
            full_metric_report_list.append(df_temp)

    full_metric_report = pd.concat(full_metric_report_list, axis=0)
    full_metric_report_summed = full_metric_report.groupby("variable").mean()

    full_metric_report.to_csv(
        os.path.join(validation_output_folder, f"full_metric_report_{htf_greenius}.csv")
    )
    full_metric_report_summed.to_csv(
        os.path.join(
            validation_output_folder, f"full_metric_report_summed_{htf_greenius}.csv"
        )
    )

    i_placement = 1

    # plot correlations
    plot_correlation(
        x=greenius_ref_dict[i_placement]["Q out"].values,
        y=out_reskit.sel(location=i_placement)["HeattoPlant_W"].values * 1e-6,
        x_label="HeattoPlant Greenius [MW]",
        y_label="HeattoPlant RESKit [MW]",
        title="Correlation Q out",
        savepath=os.path.join(
            validation_output_folder, f"corr_Q_out_{htf_greenius}.png"
        ),
    )

    plot_correlation(
        x=greenius_ref_dict[i_placement]["T HTFmean"].values,
        y=out_reskit.sel(location=i_placement)["HTF_mean_temperature_C"].values,
        x_label="HTF_mean_temperature Greenius [°C]",
        y_label="HTF_mean_temperature RESKit [°C]",
        title="Correlation T out",
        savepath=os.path.join(
            validation_output_folder, f"corr_T_out_{htf_greenius}.png"
        ),
    )

    # plot time series
    # plot two vars
    plot_time_series(
        i_placement=0,
        greenius_ref_dict=greenius_ref_dict,
        out_reskit=out_reskit,
        varnames1=["Q out", "HeattoPlant_W"],
        varnames2=["T HTFmean", "HTF_mean_temperature_C"],
        factors=[1e-6, 1],
        start=2400,
        stop=2448,
        placements=placements,
        htf_greenius=htf_greenius,
    )

    plot_time_series(
        i_placement=0,
        greenius_ref_dict=greenius_ref_dict,
        out_reskit=out_reskit,
        varnames1=["Inc.ang.", "theta"],
        varnames2=["IAM", "IAM"],
        factors=[1, 1],
        start=2400,
        stop=2448,
        placements=placements,
        htf_greenius=htf_greenius,
    )

    out_reskit["P_par_field_W"] = (
        out_reskit["Parasitics_solarfield_W_el"] - out_reskit["P_heating_W"]
    )
    plot_time_series(
        i_placement=0,
        greenius_ref_dict=greenius_ref_dict,
        out_reskit=out_reskit,
        varnames1=["QFP Aux", "P_heating_W"],
        varnames2=["W parField", "P_par_field_W"],
        factors=[1e-6, 1e-6],
        start=2400,
        stop=2448,
        placements=placements,
        htf_greenius=htf_greenius,
    )


def plot_time_series(
    i_placement,
    greenius_ref_dict,
    out_reskit,
    varnames1,
    varnames2,
    factors,
    start,
    stop,
    placements,
    htf_greenius,
):
    location = placements.loc[i_placement].Name

    x = range(start, stop)

    data_gr_heat = greenius_ref_dict[i_placement][varnames1[0]].values[start:stop]
    data_re_heat = (
        out_reskit.sel(location=i_placement)[varnames1[1]].values[start:stop]
        * factors[0]
    )

    data_gr_temp = greenius_ref_dict[i_placement][varnames2[0]].values[start:stop]
    data_re_temp = (
        out_reskit.sel(location=i_placement)[varnames2[1]].values[start:stop]
        * factors[1]
    )

    y1 = [
        data_gr_heat,
        data_re_heat,
    ]
    y1_legend = ["Greenius", "RESKit"]
    y1_axis = varnames1[1]  #'Heat from\nField [MW]'

    # lower plot
    y2 = [
        data_gr_temp,
        data_re_temp,
    ]
    y2_legend = ["Greenius", "RESKit"]
    y2_axis = varnames2[1]  # 'Mean Temperatue\nHTF [°C]'

    title = f"Time Series Comparision in \n {location}"

    savepath = os.path.join(
        validation_output_folder, f"ts_plot_{htf_greenius}_{y1_axis}_{y2_axis}.png"
    )
    plot_time_series_twovars(
        x=x,
        y1=y1,
        y1_legend=y1_legend,
        y1_axis=y1_axis,
        y2=y2,
        y2_legend=y2_legend,
        y2_axis=y2_axis,
        title=title,
        savepath=savepath,
    )


# RMSE
def RMSE(ser1, ser2):
    diff = ser1 - ser2
    return np.sqrt((diff ** 2).mean())


# absolute Error
def absError(ser1, ser2):
    diff = ser1 - ser2
    return np.abs(diff).mean()


# mean deviation
def meanDev(ser1, ser2):
    diff = ser1 - ser2
    return diff.mean()


# relative absolute error
def relabsError(ser1, ser2):
    diff = ser1 - ser2
    return np.abs(diff).mean() / max(0.01, ser2.mean())


# relative mean deviation
def relmeanDev(ser1, ser2):
    diff = ser1 - ser2
    return diff.mean() / max(0.01, ser2.mean())


# peak load
def peakLoad(ser1, ser2):
    return ser1.max(), max(0.01, ser2.mean())


# std of diff
def stdDifference(ser1, ser2):
    diff = ser1 - ser2
    return diff.std()


# correlation coefficient
def corrCoefficient(ser1, ser2):
    return np.corrcoef(ser1, ser2)[0, 1]


# time series
def timeSeries(ser1, ser2):
    return ser1 - ser2


# meanCF
def meanCF(ser1, ser2):
    return ser1.mean()


def Nash_Sutcliffe(ser1, ser2):
    # 1 - (ser1-ser2)^2 / (ser2-avg_ser2)^2
    return 1 - (((ser1 - ser2) ** 2)).sum() / (((ser2 - ser2.mean()) ** 2).sum())


def DCCA(ser1, ser2):
    try:
        # zero-mean cumulative sum
        ser1 = fu.toAggregated(ser1.to_numpy())
        ser2 = fu.toAggregated(ser2.to_numpy())

        # initialize non-empty dcca object
        pydcca = fathon.DCCA(ser1, ser2)

        # compute rho index
        wins = np.array(
            [24], dtype="int64"
        )  # only take a window size of 1 day (24h) , 7*24, 30*24, 91*24])
        # wins = fu.linRangeByStep(24, 25, step=50)
        n, rho = pydcca.computeRho(wins, polOrd=1)
        return rho[0]
    except:
        return np.NaN


def plot_correlation(x, y, x_label, y_label, title, savepath=None):

    # Font sizes:
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    MAX_SIZE = 22

    # set style to latex
    # plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": ["Helvetica"]})

    # calculate R-value
    _, _, R_value, _, _ = stats.linregress(x, y)
    R_squared = R_value ** 2

    # determine R^2 = 1 line
    line_min = min(min(x), min(y))
    line_max = max(max(x), max(y))
    step = (line_max - line_min) / 20
    line = np.arange(start=line_min, stop=line_max + step, step=step)

    # plot itself
    fig, axs = plt.subplots(dpi=600, figsize=(16 / 2.54, 9 / 2.54))
    # plot points
    axs.plot(x, y, "xk")

    # plot line
    axs.plot(
        line,
        line,
        color="grey",
        # marker='o',
        linestyle="-",
        linewidth=2,
        # markersize=12
    )

    # set lims
    axs.set_xlim((line_min, line_max + 1))
    axs.set_ylim((line_min, line_max + 1))
    # set titel and font sizes etc
    axs.set_xlabel(x_label, fontsize=SMALL_SIZE)
    axs.set_ylabel(y_label, fontsize=SMALL_SIZE)
    axs.set_title(title, fontsize=SMALL_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    axs.grid("on")

    axs.text(
        x=0.05,  # position
        y=0.95,  # position
        s=f"$R^2={round(R_squared,5)}$",  # text
        transform=axs.transAxes,  # coordinate system: trans ax: (0,0) lower left, (1,1) upper right
        horizontalalignment="left",
        verticalalignment="top",
        bbox=dict(
            edgecolor="black", facecolor="white", alpha=1
        ),  # box around the text,
        fontsize=SMALL_SIZE,  # text size
    )
    plt.tight_layout()

    if not savepath == None:
        plt.savefig(
            savepath, dpi=600, bbox="thight",
        )
        print("Figure saved to:", os.path.abspath(savepath))


def plot_time_series_twovars(
    x, y1, y1_legend, y1_axis, y2, y2_legend, y2_axis, title, savepath=None
):

    # Font sizes:
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    MAX_SIZE = 22

    # set style to latex
    # plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": ["Helvetica"]})

    assert len(y1) == len(y1_legend)
    assert len(y2) == len(y2_legend)

    # plot
    # plot itself
    fig, axs = plt.subplots(2, 1, sharex=True, dpi=600, figsize=(13 / 2.54, 12 / 2.54))

    # loop all y data
    for y_data, y_label, color in zip(y1, y1_legend, colors):
        axs[0].plot(
            x,
            y_data,
            color=color,
            # marker='o',
            linestyle="-",
            linewidth=2,
            # markersize=12,
            label=y_label,
        )

    for y_data, y_label, color in zip(y2, y2_legend, colors):
        axs[1].plot(
            x,
            y_data,
            color=color,
            # marker='o',
            linestyle="-",
            linewidth=2,
            # markersize=12,
            label=y_label,
        )

    # set atrributs upper plot
    axs[0].set_xlim((min(x), max(x)))
    axs[0].set_ylabel(y1_axis, fontsize=SMALL_SIZE)
    axs[0].set_title(title, fontsize=SMALL_SIZE)
    # set attributs lower plot
    # axs.set_ylim((line_min, line_max+1))
    # set titel and font sizes etc
    axs[1].set_xlabel("time since 01.01.2015, 00:00 [h]", fontsize=SMALL_SIZE)
    axs[1].set_ylabel(y2_axis, fontsize=SMALL_SIZE)

    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    axs[0].grid("on")
    axs[1].grid("on")
    axs[0].legend()

    plt.tight_layout()

    if not savepath == None:
        plt.savefig(
            savepath, dpi=600, bbox="thight",
        )
        print("Figure saved to:", os.path.abspath(savepath))


colors = [
    "#023D6B",
    "#ADBDE3",
    "#6D268E",
    "#30A93B",
    "#FFE900",
    "#FF8C0C",
    "#DF0F44",
]

if __name__ == "__main__":
    main(
        datasetname=datasetname, validation_output_folder=validation_output_folder,
    )
