import gdown

from dataset import ECGDatasetUpdate, DEVICE, ECGDataset200ms, ECGDatasetRandomStart
from import_data import train_featurevector, val_featurevector
from torch.utils.data import Dataset, DataLoader
from lstm import LSTM
from trainloop import Trainer
from lstm_conv import LSTM_Conv
from lstm_2layers_stacked import LSTM_2stacked
#from trainloop import val_bacc
from transformer2 import Transformer
import torch
from torch.utils.data import DataLoader
from trainloop import Trainer
from lstm_2layers_stacked import LSTM_2stacked
from torch.utils.tensorboard import SummaryWriter  

writer = SummaryWriter()


# Define the path to your checkpoint
url = 'https://drive.google.com/uc?id=1-0Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z'
output = "checkpoint.pth"
gdown.download(url, output, quiet=False)

# Initialize the model and optimizer
model = LSTM_2stacked().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())

# Load the checkpoint
checkpoint = torch.load(output)

# Load the state dict into the model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# If the checkpoint includes the epoch number, load it
start_epoch = checkpoint.get('epoch', 0)

# Continue with the rest of your main.py code
train_loader = DataLoader(ECGDatasetUpdate, batch_size=32, shuffle=True)
val_loader = DataLoader(ECGDatasetUpdate, batch_size=32, shuffle=False)

loss = torch.nn.CrossEntropyLoss()
trainer = Trainer(model, loss, writer)

# Start the epochs from where the checkpoint left off
for epoch in range(start_epoch, 100):
    # Training phase
    trainer.epoch(train_loader, True, epoch)

    # Validation phase
    trainer.epoch(val_loader, False, epoch)