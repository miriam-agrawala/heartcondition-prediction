# Import necessary libraries
import os
import random
import math
import torch
import wfdb
import tqdm
from import_data import path_data_csv

# Set the device to CUDA if available else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

# Define a class for the ECG dataset with full 10 second sequences
class ECGDatasetUpdate(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data # Input data
        self.ecg = None # Placeholder for ECG signals
        self.labels = None # Placeholder for labels
            
        N  = self.data.shape[0] # Number of datapoints
        self.ecg = torch.empty(N, 1000, 12).to(DEVICE) # Initialize tensor for ECG signals
        self.labels = torch.empty(N, 1).type(torch.long).to(DEVICE) # Initialize tensor for labels

        # Loop over each datapoint
        for idx in tqdm.tqdm(range(N)):
            # Read the ECG signal from the file 
            path_wfdb_100 = os.path.join(path_data_csv, self.data[idx][4])
            record_100 = wfdb.rdrecord(path_wfdb_100)
            signal_physical_100 = record_100.p_signal
            # Store the ECG signal and the label          
            self.ecg[idx,:,:] = torch.Tensor(signal_physical_100).to(DEVICE).unsqueeze(0)
            self.labels[idx,:] = torch.Tensor([self.data[idx][7]]).type(torch.long).to(DEVICE)

    def __len__(self):
        # Return the number of datapoints
        return self.data.shape[0]
  
    def __getitem__(self, idx):
        # Return the ECG signal and the label for a given index
        return self.ecg[idx, :, :], self.labels[idx]

# Define a class for ECG dataset with last 200 datapoints (of 1000)
class ECGDataset200(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data # Input data
        self.ecg = None # Placeholder for ECG signals
        self.labels = None # Placeholder for labels
          
        N  = self.data.shape[0] # Number of datapoints
        self.ecg = torch.empty(N, 200, 12).to(DEVICE) # Initialize tensor for ECG signals
        self.labels = torch.empty(N, 1).type(torch.long).to(DEVICE) # Initialize tensor for labels

        # Loop over each datapoint
        for idx in tqdm.tqdm(range(N)):
            # Read the ECG signal from the file
            path_wfdb_100 = os.path.join(path_data_csv, self.data[idx][4])
            record_100 = wfdb.rdrecord(path_wfdb_100)
            signal_physical_100 = record_100.p_signal
            signal_physical_100 = signal_physical_100[-200:] # Take the last 200 datapoints
            # Store the ECG signal and the label    
            self.ecg[idx,:,:] = torch.Tensor(signal_physical_100).to(DEVICE).unsqueeze(0)
            self.labels[idx,:] = torch.Tensor([self.data[idx][7]]).type(torch.long).to(DEVICE)

    def __len__(self):
        # Return the number of datapoints
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Return the ECG signal and the label for a given index
        return self.ecg[idx, :, :], self.labels[idx]

# Define a class for ECG dataset with random start point  
class ECGDatasetRandomStart(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data # Input data
        self.ecg = None # Placeholder for ECG signals
        self.labels = None # Placeholder for labels

        N  = self.data.shape[0] # Number of datapoints
        self.ecg = torch.empty(N, 1000, 12).to(DEVICE) # Initialize tensor for ECG signals
        self.labels = torch.empty(N, 1).type(torch.long).to(DEVICE) # Initialize tensor for labels

        # Loop over each datapoint
        for idx in tqdm.tqdm(range(N)):
            # Read the ECG signal from the file    
            path_wfdb_100 = os.path.join(path_data_csv, self.data[idx][4])
            record_100 = wfdb.rdrecord(path_wfdb_100)
            signal_physical_100 = record_100.p_signal
            # Store the ECG signal and the label         
            self.ecg[idx,:,:] = torch.Tensor(signal_physical_100).to(DEVICE).unsqueeze(0)
            self.labels[idx,:] = torch.Tensor([self.data[idx][7]]).type(torch.long).to(DEVICE)

    def __len__(self):
        # Return the number of datapoints
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Generate a random start point
        start = math.floor(random.uniform(0,200))
        # Return the ECG signal from the random start point and the label for a given index
        return self.ecg[idx, start:(start+800), :], self.labels[idx]

