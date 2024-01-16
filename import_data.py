
import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
import os
import json
import numpy as np


np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

#path_data_csv = r"C:\Users\magra\Documents\HSD\5_Semester\Advances_AI\Project\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
#path_data_csv = r"C:\Users\magra\Documents\5_Semester\Advances_AI\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
path_data_csv =r"/home/miriam/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

ptbxl = pd.read_csv(os.path.join(path_data_csv, "ptbxl_database.csv"))
print(ptbxl.head(5))

scp_codes = pd.read_csv(os.path.join(path_data_csv, "scp_statements.csv"))
print(scp_codes.head(5))

ptbxl_ann = ptbxl[["ecg_id", "patient_id", "scp_codes", "filename_lr"]]

intermediary_store=[]
for index, row in ptbxl_ann.iterrows():
    row["scp_codes"] = row["scp_codes"].replace("'", '"')
    row["scp_codes"] = json.loads(row["scp_codes"])
    intermediary_store.append(row["scp_codes"])

ptbxl_ann["scp_codes"] = intermediary_store

data_one_scp_code = []
for index, row in ptbxl_ann.iterrows():
    dictionary = row["scp_codes"]
    
    
    if isinstance(dictionary, dict):
        #print(dictionary)
        # find key that corresponds to highest value
        max_key = max(dictionary, key = dictionary.get)
        #print(max_key)
        # get value associated with max_key
        max_value = dictionary[max_key]

        if max_value >= 70:
            # first two columns stay the same
            ecg_id_value = row["ecg_id"]
            patient_id_value = row["patient_id"]
            filename_100 = row["filename_lr"]
            # add to list for storage
            data_one_scp_code.append([ecg_id_value, patient_id_value, max_key, max_value, filename_100])

data_one_scp_df = pd.DataFrame(data_one_scp_code, columns=["ecg_id", "patient_id", "max_key", "max_value", "filename_lr"])

diagnostic_class = []
diagnostic_subclass = []
for i in data_one_scp_df["max_key"]:
    #print("i", i)
    diagnostic_class.append(scp_codes.loc[scp_codes["Unnamed: 0"] == i, "diagnostic_class"].values[0])
    diagnostic_subclass.append(scp_codes.loc[scp_codes["Unnamed: 0"] == i, "diagnostic_subclass"].values[0])

data_one_scp_df["diagnostic_class"] = diagnostic_class
data_one_scp_df["diagnostic_subclass"] = diagnostic_subclass

data_one_scp_df=data_one_scp_df.dropna()

data_one_scp_df["diagnostic_class_numerical"] = data_one_scp_df.loc[:, "diagnostic_class"]

data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "NORM", "diagnostic_class_numerical"] = 0
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "HYP", "diagnostic_class_numerical"] = 1
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "STTC", "diagnostic_class_numerical"] = 2
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "CD", "diagnostic_class_numerical"] = 3
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "MI", "diagnostic_class_numerical"] = 4

print(data_one_scp_df.head(5))

prelim_featurevector = pd.DataFrame(data_one_scp_df).to_numpy()

# train/val split

from sklearn.model_selection import train_test_split

unique_patient_ids = data_one_scp_df["patient_id"].unique()
train_ids, val_ids = train_test_split(unique_patient_ids, test_size=0.2, random_state=0)

train_df = data_one_scp_df[data_one_scp_df["patient_id"].isin(train_ids)]
val_df = data_one_scp_df[data_one_scp_df["patient_id"].isin(val_ids)]

print(f"Training set size: {train_df.shape}")
print(f"Validation set size: {val_df.shape}")

train_featurevector = pd.DataFrame(train_df).to_numpy()
val_featurevector = pd.DataFrame(val_df).to_numpy()