from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import wfdb
import tqdm
import os
from import_data import path_data_csv, prelim_featurevector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

class ECGDatasetUpdate(torch.utils.data.Dataset):
  def __init__(self, seqlen):
    self.data = prelim_featurevector
 
    self.ecg = None
    self.labels = None
    self.seqlen = seqlen
   
    N  = self.data.shape[0]
    self.ecg = torch.empty(N, 1000, 12).to(DEVICE)
    self.labels = torch.empty(N, 1).type(torch.long).to(DEVICE)
 
    for idx in tqdm.tqdm(range(N)):    
        path_wfdb_100 = os.path.join(path_data_csv, prelim_featurevector[idx][4])
        record_100 = wfdb.rdrecord(path_wfdb_100)
        signal_physical_100 = record_100.p_signal  
       
        self.ecg[idx,:,:] = torch.Tensor(signal_physical_100).to(DEVICE).unsqueeze(0)
        self.labels[idx,:] = torch.Tensor([prelim_featurevector[idx][7]]).type(torch.long).to(DEVICE)

  def __len__(self):
    return self.data.shape[0]
   # return len(self.data) - self.seqlen
  
  def __getitem__(self, idx):
    #for idx in tqdm.tqdm(range(100)):
    return self.ecg[idx, :, :], self.labels[idx]
    #return self.ecg[idx:(idx+self.seqlen)], self.labels[idx+self.seqlen]