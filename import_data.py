# import libraries
import os
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Set numpy print options to display all elemnts of an array
np.set_printoptions(threshold=np.inf)

# Set pandas options to display all rows and columns of a DataFrame
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Define the path to the CSV data
path_data_csv = r"C:\Users\magra\Documents\HSD\5_Semester\Advances_AI\Project\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
#path_data_csv = r"/home/miriam/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
#path_data_csv = r"home/miriamagrawala/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

# Load the ptbxl_database.csv file into a DataFrame
ptbxl = pd.read_csv(os.path.join(path_data_csv, "ptbxl_database.csv"))
print(ptbxl.head(5))

# Load the scp_statements.csv file into a DataFrame
scp_codes = pd.read_csv(os.path.join(path_data_csv, "scp_statements.csv"))
print(scp_codes.head(5))

# Select specific columns from the ptbxl DataFrame
ptbxl_ann = ptbxl[["ecg_id", "patient_id", "scp_codes", "filename_lr"]]

# Modify the scp_codes column to be a list of dictionaries
# Initialize an empty list to store scp_codes after processing
intermediary_store=[]
# Iterate over each row in the DataFrame
for index, row in ptbxl_ann.iterrows():
    # Replace single quotes with double quotes in the scp_codes string
    row["scp_codes"] = row["scp_codes"].replace("'", '"')
    # Convert the scp_codes string to a dictionary
    row["scp_codes"] = json.loads(row["scp_codes"])
    # Append the dictionary to the list
    intermediary_store.append(row["scp_codes"])

# Replace the scp_codes column in the DataFrame with the processed list
ptbxl_ann["scp_codes"] = intermediary_store

# Keep only the scp codes with a confidence level of 70 or higher
# Initialize an empty list to store data with one scp_code
data_one_scp_code = []
# Iterate over each row in the DataFrame
for index, row in ptbxl_ann.iterrows():
    # Get the scp_codes dictionary
    dictionary = row["scp_codes"]

    # Check if the scp_codes is a dictionary
    if isinstance(dictionary, dict):
        # Find key that corresponds to highest value
        max_key = max(dictionary, key = dictionary.get)
        # Get value associated with max_key
        max_value = dictionary[max_key]

        # Check if the max_value is greater than or equal to 70
        if max_value >= 70:
            # Get the values of the first two columns
            ecg_id_value = row["ecg_id"]
            patient_id_value = row["patient_id"]
            filename_100 = row["filename_lr"]
            # Append the values to the list
            data_one_scp_code.append([ecg_id_value, patient_id_value, max_key, max_value, filename_100])

# Convert the list to a DataFrame
data_one_scp_df = pd.DataFrame(data_one_scp_code, columns=["ecg_id", "patient_id", "max_key", "max_value", "filename_lr"])

# Initialize two empty lists to store diagnostic_class and diagnostic_subclass
diagnostic_class = []
diagnostic_subclass = []
# Iterate over each max_key in the DataFrame
for i in data_one_scp_df["max_key"]:
    # Append the corresponding diagnostic_class and diagnostic_subclass to the lists
    diagnostic_class.append(scp_codes.loc[scp_codes["Unnamed: 0"] == i, "diagnostic_class"].values[0])
    diagnostic_subclass.append(scp_codes.loc[scp_codes["Unnamed: 0"] == i, "diagnostic_subclass"].values[0])

# Add the diagnostic_class and diagnostic_subclass lists as new columns in the DataFrame
data_one_scp_df["diagnostic_class"] = diagnostic_class
data_one_scp_df["diagnostic_subclass"] = diagnostic_subclass

# Drop rows with missing values
data_one_scp_df=data_one_scp_df.dropna()

# Add a new column diagnostic_class_numerical with the same values as diagnostic_class
data_one_scp_df["diagnostic_class_numerical"] = data_one_scp_df.loc[:, "diagnostic_class"]

# Replace string values in the diagnostic_class_numerical column with numerical values
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "NORM", "diagnostic_class_numerical"] = 0
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "HYP", "diagnostic_class_numerical"] = 1
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "STTC", "diagnostic_class_numerical"] = 2
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "CD", "diagnostic_class_numerical"] = 3
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "MI", "diagnostic_class_numerical"] = 4
print(data_one_scp_df.head(5))

# Convert the DataFrame to a numpy array
prelim_featurevector = pd.DataFrame(data_one_scp_df).to_numpy()

# Get unique patient_ids to perform the train/test split
unique_patient_ids = data_one_scp_df["patient_id"].unique()
# Split the patient_ids into training and validation sets
train_ids, val_ids = train_test_split(unique_patient_ids, test_size=0.2, random_state=0)

# Get the rows in the DataFrame that have patient_id in the training set
train_df = data_one_scp_df[data_one_scp_df["patient_id"].isin(train_ids)]
# Get the rows in the DataFrame that have patient_id in the validation set
val_df = data_one_scp_df[data_one_scp_df["patient_id"].isin(val_ids)]
# Print the size of the training and validation sets
print(f"Training set size: {train_df.shape}")
print(f"Validation set size: {val_df.shape}")

# Convert the training and validation DataFrames to numpy arrays
train_featurevector = pd.DataFrame(train_df).to_numpy()
val_featurevector = pd.DataFrame(val_df).to_numpy()

