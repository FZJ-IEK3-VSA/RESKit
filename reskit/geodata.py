#%%
import geokit as gk
import os
import pandas as pd
from workflow_manager import WorkflowManager

abs_path = os.getcwd()
file_name = (
    "glaes_result_placements__Onshore__Argentina__ARG.10_1__default2__SG453-454.shp"
)
full_path = os.path.join(abs_path, "reskit", file_name)
# full_path = os.path.join(abs_path, file_name)
print("Path: ", full_path)


df_geom = gk.vector.extractFeatures(full_path)
print(df_geom, type(df_geom))


# %%
wfm = WorkflowManager(df_geom)
print("done")

# %%
