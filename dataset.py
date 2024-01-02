from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import wfdb
import tqdm
import os
from import_data import path_data_csv, prelim_featurevector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

class ECGDataset(torch.utils.data.Dataset):
  def __init__(self, seqlen):
    self.data = prelim_featurevector

    self.ecg = None
    self.labels = None
    self.seqlen = seqlen
    
    for idx in tqdm.tqdm(range(self.data.shape[0])):
    #for idx in tqdm.tqdm(range(100)):    

        path_wfdb_100 = os.path.join(path_data_csv, prelim_featurevector[idx][4])
        record_100 = wfdb.rdrecord(path_wfdb_100)
        signal_physical_100 = record_100.p_signal  
        #print("signal physical", signal_physical_100.shape)

        ecg = torch.Tensor(signal_physical_100).to(DEVICE).unsqueeze(0) #.view(1,-1) (permute)
        #ecg = ecg[:, -200:, :]
        #print("ecg", ecg.shape)
        
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
    #for idx in tqdm.tqdm(range(100)):
    return self.ecg[idx, :, :], self.labels[idx]
    #return self.ecg[idx:(idx+self.seqlen)], self.labels[idx+self.seqlen]