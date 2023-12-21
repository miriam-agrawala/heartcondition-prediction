
# %%
import wfdb
import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
import os
import json
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

# %%

np.set_printoptions(threshold=np.inf)

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

path_data_csv = r"C:\Users\magra\Documents\HSD\5_Semester\Advances_AI\Project\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
# path_data_csv = r"C:\Users\magra\Documents\5_Semester\Advances_AI\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ptbxl = pd.read_csv(os.path.join(path_data_csv, "ptbxl_database.csv"))
ptbxl.head()

scp_codes = pd.read_csv(os.path.join(path_data_csv, "scp_statements.csv"))



ptbxl_ann = ptbxl[["ecg_id", "patient_id", "scp_codes", "filename_lr"]]

intermediary_store=[]
import json
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

for index, row in data_one_scp_df.iterrows():
    print(data_one_scp_df["max_key"])

# %%
# %%
diagnostic_class = []
diagnostic_subclass = []
for i in data_one_scp_df["max_key"]:
    #print("i", i)
    diagnostic_class.append(scp_codes.loc[scp_codes["Unnamed: 0"] == i, "diagnostic_class"].values[0])
    diagnostic_subclass.append(scp_codes.loc[scp_codes["Unnamed: 0"] == i, "diagnostic_subclass"].values[0])
    

# %%
print(data_one_scp_df.shape)
print(len(diagnostic_class))
print(len(diagnostic_subclass))
print(set(diagnostic_class))

# %%
data_one_scp_df["diagnostic_class"] = diagnostic_class
data_one_scp_df["diagnostic_subclass"] = diagnostic_subclass

# %%
print(scp_codes.loc[scp_codes["Unnamed: 0"] == "AFLT"])

# %%
data_one_scp_df[data_one_scp_df["diagnostic_class"].isnull()].shape 

# %%
data_one_scp_df[data_one_scp_df["diagnostic_subclass"].isnull()].shape

# %%
(data_one_scp_df["diagnostic_subclass"].isnull() == data_one_scp_df["diagnostic_class"].isnull()).unique

# %%
data_one_scp_df.shape

# %%
data_one_scp_df=data_one_scp_df.dropna()

# %%
data_one_scp_df.shape

# %%
data_one_scp_df.head(10)

# %%
data_one_scp_df["diagnostic_class_numerical"] = data_one_scp_df.loc[:, "diagnostic_class"]
data_one_scp_df.head(5)

# %%
data_one_scp_df["diagnostic_class"].unique()

# %%
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "NORM", "diagnostic_class_numerical"] = 0
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "HYP", "diagnostic_class_numerical"] = 1
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "STTC", "diagnostic_class_numerical"] = 2
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "CD", "diagnostic_class_numerical"] = 3
data_one_scp_df.loc[data_one_scp_df["diagnostic_class_numerical"] == "MI", "diagnostic_class_numerical"] = 4
data_one_scp_df.head(20)
#'HYP', 'STTC', 'CD', 'NORM', 'MI

# %%
data_one_scp_df["diagnostic_class_numerical"].unique()

# %%
prelim_featurevector = pd.DataFrame(data_one_scp_df).to_numpy()
print(prelim_featurevector)
print(type(prelim_featurevector))
print(prelim_featurevector.shape)

# %%
prelim_featurevector_df = pd.DataFrame(prelim_featurevector)
#prelim_featurevector_df
#1 if prelim_featurevector_df[2].loc[0] == 'NORM' else 0




# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

# %%
class ECGDataset(torch.utils.data.Dataset):
  def __init__(self, seqlen):
    self.data = prelim_featurevector

    self.ecg = None
    self.labels = None
    self.seqlen = seqlen
    
    for idx in tqdm.tqdm(range(self.data.shape[0])):

        path_wfdb_100 = os.path.join(path_data_csv, prelim_featurevector[idx][4])
        record_100 = wfdb.rdrecord(path_wfdb_100)
        signal_physical_100 = record_100.p_signal  

        ecg = torch.Tensor(signal_physical_100).to(DEVICE).unsqueeze(0) #.view(1,-1) (permute)
        
        #self.ecg.append(signal_physical_100)

        if self.ecg is None:
          self.ecg = ecg
        else:
          self.ecg = torch.concat((self.ecg, ecg), dim=0)
        #print(prelim_featurevector[idx][7])
        #label = torch.Tensor(prelim_featurevector[idx][-1]).type(torch.long).to(DEVICE)
        label = torch.Tensor([prelim_featurevector[idx][7]]).type(torch.long).to(DEVICE)
        #self.labels.append(prelim_featurevector[idx][2])
        
        if self.labels is None:
          self.labels = label
        else:
          self.labels = torch.concat((self.labels, label), dim=0)

    #self.ecg = torch.tensor(self.ecg)
    #self.labels = torch.tensor(self.labels)


  def __len__(self):
    return self.data.shape[0]
   # return len(self.data) - self.seqlen
  
  def __getitem__(self, idx):
    return self.ecg[idx, :, :], self.labels[idx]
    #return self.ecg[idx:(idx+self.seqlen)], self.labels[idx+self.seqlen]

# %%
dataset = ECGDataset(seqlen=4)

# %%
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
print(next(iter(dataloader)))

# %%
batch = next(iter(dataloader))

# %%
len(batch)

# %%
batch[0].shape

# %%
batch[1]

# %%
batch[1].shape

# %%
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU() 

    def forward(self,x):
        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

# %%
len(dataset)

# %%


# %%
torch.set_printoptions(profile="full")
#print(dataset.labels)
print(dataset.labels.size())

# %%
for i in range (29):
    print(dataset[i][1])

# %%
dataset[0][0].shape

# %%
dataset[0][0]

# %%
dataset[0][1]

# %%
path_wfdb = r"C:\Users\magra\Documents\HSD\5_Semester\Advances_AI\Project\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
#path_wfdb = path_wfdb.replace('\\', "/")
path_wfdb

# %%
wfdb_data = []
for i in tqdm.tqdm(range(prelim_featurevector.shape[0])):
    #print(prelim_featurevector[i][4])
    path_wfdb_100 = os.path.join(path_data_csv,  prelim_featurevector[i][4])
    #path_wfdb_100
    record_100 = wfdb.rdrecord(path_wfdb_100)
    signal_physical_100 = record_100.p_signal
    
    wfdb_data.append(signal_physical_100)
    #print(wfdb_data)

# %%
wfdb_data

# %%
wfdb_data[0][0][0]

# %%
wfdb_data[0]

# %%
type(wfdb_data[0])

# %%
wfdb_data_np = np.array(wfdb_data)
print(wfdb_data_np[0].shape)


# %%
wfdb_data_np.shape[0]



# %%
class WaveData(Dataset):
    def __init__(self):
        # data loading
        pass
    
    def __getitem__(self, index):
        # allows indexing
        pass

    def __len__(self):
        # len(dataset)
        pass

# %%
prelim_featurevector_df["wfdb"] = wfdb_data_np.tolist()

# %%
prelim_featurevector_df

# %%
prelim_featurevector_wfdb_np = prelim_featurevector_df.to_numpy()

# %%
prelim_featurevector_wfdb_np.shape

# %%
prelim_featurevector_wfdb_np

# %%
#wfdb_data_np = wfdb_data_np.reshape(19683, -1)
prelim_featurevector_wfdb = np.concatenate((prelim_featurevector, wfdb_data_np[:,:,:].reshape((19683, -1))), axis=1)

# %%
prelim_featurevector_wfdb = np.hstack((prelim_featurevector, wfdb_data_np))

# %%
prelim_featurevector_wfdb.shape

# %%
print(path_wfdb)
path_wfdb_100 = os.path.join(path_wfdb,  prelim_featurevector[0][4])
path_wfdb_100

# %%
record = wfdb.rdrecord(path_wfdb_100)
wfdb.plot_wfdb(record=record, title='Trial, ECG, Ending .hea') 
display(record.__dict__)

# %%
signal_physical = record.p_signal
print(signal_physical)
signal_physical.shape


