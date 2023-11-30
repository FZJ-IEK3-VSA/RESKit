import geokit as gk
import pandas as pd
from pandas.core.frame import DataFrame
import reskit as rk
import time
import numpy as np
import os
from reskit.csp.data import csp_data_path


class dataset_handler:
    def __init__(self, datasets) -> None:
        """dataset_handelr, which applies usefulll functions for splitting up placements from different datasets

        Parameters
        ----------
        datasets : list of strings
            each string should be a valid key in the
        """
        assert isinstance(datasets, list)

        self.datasets = datasets

    def split_placements(self, placements, gsa_dni_path, gsa_tamb_path):

        assert "lat" in placements.columns
        assert "lon" in placements.columns

        placements["geom"] = placements[["lon", "lat"]].apply(
            lambda x: gk.geom.point(x[0], x[1], srs=4326), axis=1
        )

        placements["dni_gsa"] = (
            gk.raster.interpolateValues(source=gsa_dni_path, points=placements.geom)
            * 1000
            / 24
        )

        placements["tamb_gsa"] = gk.raster.interpolateValues(
            source=gsa_tamb_path, points=placements.geom
        )

        mat_HTF_opt = self._get_opt_HTF_matrix()

        placements["Dataset_opt"] = placements[["dni_gsa", "tamb_gsa"]].apply(
            lambda x: self._lookup(x[0], x[1], mat_HTF_opt), axis=1
        )
        placements = placements.drop(["geom", "dni_gsa", "tamb_gsa"], axis=1)

        return placements

    def _get_path_dataset_opt(self):

        datasets = self.datasets

        def _list_to_str(li):
            s = ""
            for l in li:
                l = l.replace("Dataset_", "").split("_")[0]
                if s == "":
                    s = l
                else:
                    s += "_" + l
            return s

        path = os.path.join(
            csp_data_path, f"optimal_htf_selection{_list_to_str(datasets)}.csv"
        )
        return path

    def _get_opt_HTF_matrix(self):
        """tries to find the opt matrix from the given datasets. if not possible, calculate a new one

        Returns
        -------
        pd.DataFrame
            lookup table with the optimal HTF fluid for each temperature and DNI
        """
        path = self._get_path_dataset_opt()
        if os.path.isfile(path):
            htf_opt_matrix = pd.read_csv(path, index_col=[0], header=[0])
            htf_opt_matrix.index = htf_opt_matrix.index.astype(float).astype(int)
            htf_opt_matrix.columns = htf_opt_matrix.columns.astype(float).astype(int)
        else:
            print("No opt HTF matrix found. Calculating new Matrix.")
            htf_opt_matrix = self._calc_opt_HTF_matrix()
            htf_opt_matrix.index = htf_opt_matrix.index.astype(int)
            htf_opt_matrix.columns = htf_opt_matrix.columns.astype(int)
        return htf_opt_matrix

    def _calc_opt_HTF_matrix(self) -> pd.DataFrame:
        """calculates the optimal htf for a variation of t_amb and dni and stores it inside reskit

        Returns
        -------
        [pd.DataFrame]
            [matrix with opt htf for different T_amb and DNIs]
        """
        dT_vector = np.arange(-40, 30, 10)
        fDNI_vector = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        dT_matrix = np.tile(dT_vector, (len(fDNI_vector), 1))
        fDNI_matrix = np.tile(fDNI_vector, (len(dT_vector), 1)).T
        variable_name = "lcoe_EURct_per_kWh_el"

        n_placements = fDNI_matrix.size

        # noor 2 ptr plant, morocco
        placements = pd.DataFrame()
        placements["lon"] = [-6.8] * n_placements  # Longitude
        placements["lat"] = [31.0] * n_placements  # Latitude
        placements["area"] = [1e6] * n_placements
        placements["T_offset_K"] = dT_matrix.flatten()
        placements["DNI_factor"] = fDNI_matrix.flatten()

        print("placements:", len(placements))

        # era5_path=r'C:\Users\d.franzmann\data\ERA5\7\6'
        era5_path = r"/storage/internal/data/gears/weather/ERA5/processed/4/7/6/2015"
        datasetnames = self.datasets
        global_solar_atlas_dni_path = "default_cluster"

        outs = {}
        for datasetname in datasetnames:
            print("datasetname", datasetname)

            out = rk.csp.workflows.workflows.CSP_PTR_ERA5_specific_dataset(
                placements=placements,
                era5_path=era5_path,
                global_solar_atlas_dni_path=global_solar_atlas_dni_path,
                datasetname=datasetname,
                return_self=True,
                JITaccelerate=False,
                verbose=True,
                debug_vars=False,
                onlynightuse=True,
                fullvariation=False,
                # output_variables=['lcoe_EURct_per_kWh_el']
            )
            out = out.placements[["mean_T_amb_K", "mean_DNI_W_per_m2", variable_name]]

            outs[datasetname] = out

        # make 2D matrix
        pivots = {}
        for key in outs:
            pivots[key] = outs[key].pivot(
                index="mean_T_amb_K", columns="mean_DNI_W_per_m2", values=variable_name
            )
            pivots[key][pivots[key] < 0] = 1e9
            pivots[key][np.isinf(pivots[key])] = 1e9

        # compare
        def _getvaluedf(value, like):
            out = like.copy()
            out[:] = value
            return out

        # empty dummy string
        first_entry = list(pivots.keys())[0]
        argmins = _getvaluedf("f", pivots[first_entry])
        # check each dataset if it is min
        for key_comp in pivots:
            ismin = _getvaluedf(True, pivots[first_entry])

            # compare element comp_key with all other keys
            for key_ref in pivots:
                # skip itself
                if key_comp == key_ref:
                    continue
                isnotmin = (
                    pivots[key_ref] <= pivots[key_comp]
                )  # entries which are not min
                ismin[isnotmin] = False

            # set values in argmin
            argmins[ismin] = key_comp
        print(argmins)
        print(argmins != "f")
        assert (argmins != "f").all().all()

        argmins.to_csv(self._get_path_dataset_opt())

        return argmins

        data_heliosol = outs["Dataset_Heliosol_2030"]
        data_solarsalt = outs["Dataset_SolarSalt_2030"]
        data_therminol = outs["Dataset_Therminol_2030"]

        hel_exists = False
        sol_exists = False
        the_exists = False

        if "data_heliosol" in locals():
            hel_exists = True
        if "data_solarsalt" in locals():
            sol_exists = True
        if "data_therminol" in locals():
            the_exists = True

        variable_name = "lcoe_EURct_per_kWh_el"
        if hel_exists:
            pivot_Heliosol = data_heliosol.pivot(
                index="mean_T_amb_K", columns="mean_DNI_W_per_m2", values=variable_name
            )
        if sol_exists:
            pivot_SolarSalt = data_solarsalt.pivot(
                index="mean_T_amb_K", columns="mean_DNI_W_per_m2", values=variable_name
            )
            pivot_SolarSalt[pivot_SolarSalt < 0] = 1e9
            pivot_SolarSalt[np.isinf(pivot_SolarSalt)] = 1e9
        if the_exists:
            pivot_Therminol = data_therminol.pivot(
                index="mean_T_amb_K", columns="mean_DNI_W_per_m2", values=variable_name
            )

        # get min matrixes for each HTF
        # heliosol
        if hel_exists and sol_exists:
            min_Heliosol = pivot_Heliosol < pivot_SolarSalt
        if hel_exists and the_exists:
            if hel_exists and sol_exists:
                min_Heliosol = min_Heliosol & (pivot_Heliosol < pivot_Therminol)
            else:
                min_Heliosol = pivot_Heliosol < pivot_Therminol
        # heliosol
        if hel_exists and sol_exists:
            min_SolarSalt = pivot_SolarSalt < pivot_Heliosol
        if hel_exists and the_exists:
            if hel_exists and sol_exists:
                min_SolarSalt = min_SolarSalt & (pivot_SolarSalt < pivot_Therminol)
            else:
                min_SolarSalt = pivot_SolarSalt < pivot_Therminol

        # therminol
        if the_exists and sol_exists:
            min_Therminol = pivot_Therminol < pivot_SolarSalt
        if the_exists and hel_exists:
            if the_exists and sol_exists:
                min_Therminol = min_Therminol & (pivot_Therminol < pivot_Heliosol)
            else:
                min_Therminol = pivot_Therminol < pivot_Heliosol

        # min = pivot_SolarSalt * min_SolarSalt \
        #     + pivot_Heliosol * min_Heliosol\
        #     + pivot_Therminol * min_Therminol

        def _getstringdf(str, like):
            out = like.copy()
            out[:] = str
            return out

        if hel_exists:
            min_Heliosol = _getstringdf("H", min_Heliosol) * min_Heliosol
        if sol_exists:
            min_SolarSalt = _getstringdf("S", min_SolarSalt) * min_SolarSalt
        if the_exists:
            min_Therminol = _getstringdf("T", min_Therminol) * min_Therminol

        if hel_exists and sol_exists and the_exists:
            argmin = min_SolarSalt + min_Heliosol + min_Therminol
        elif not hel_exists:
            argmin = min_SolarSalt + min_Therminol
        elif not sol_exists:
            argmin = min_Heliosol + min_Therminol
        elif not the_exists:
            argmin = min_Heliosol + min_Therminol

        argmin.to_csv(self._get_path_htf_opt())

        return argmin

    def _lookup(self, x, y, table):

        x_index = min(table.columns, key=lambda c: abs(c - x))
        y_index = min(table.index, key=lambda i: abs(i - y))

        return table.loc[y_index, x_index]


if __name__ == "__main__":

    datasetnames = [
        "Dataset_Heliosol_2030",
        "Dataset_SolarSalt_2030",
    ]  # , 'Dataset_Therminol_2030']

    d = dataset_handler(datasets=datasetnames)
    htf_opt_matrix = d._get_opt_HTF_matrix()

    placements = pd.DataFrame()
    n_placements = 2
    placements["tamb_gsa"] = [-6.8] * n_placements  # Longitude
    placements["dni_gsa"] = [310.0] * n_placements  # Latitude
    placements["area"] = [1e6] * n_placements

    pass
