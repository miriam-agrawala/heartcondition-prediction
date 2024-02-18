# import necessary libraries
import gdown
import torch
from torch.utils.data import Dataset, DataLoader

# Import Tensorboard writer for logging
from torch.utils.tensorboard import SummaryWriter  

# Import custom modules
from dataset import DEVICE, ECGDatasetUpdate, ECGDataset200, ECGDatasetRandomStart
from trainloop import Trainer
from import_data import train_featurevector, val_featurevector
from lstm import LSTM
from lstm_2layers_stacked import LSTM_2stacked
from lstm_3layers_stacked import LSTM_3stacked
from lstm_conv import LSTM_Conv

# Initialize the tensorboard writer
writer = SummaryWriter()

# Define the URL to the checkpoint file on Google Drive
url = 'https://drive.google.com/file/d/1RJPRdnmoYSN-vt_TqVuxuLGph-kIavdp/view?usp=drive_link'
output = "checkpoint.pth"
# Download the checkpoint file using gdown
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

print("Checkpointfile downloaded")

# Initialize the model and move it to the device (CPU or GPU)
model = LSTM_Conv().to(DEVICE)
# Initialize the optimizer with the model's parameters
optimizer = torch.optim.AdamW(model.parameters())

print("Model and Optimizer initialized")

# Load the checkpoint
checkpoint = torch.load(output)

print("Checkpoint loaded")

# Load the model and optimizer states from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("Model and Optimizer loaded")

# If the checkpoint includes the epoch number, load it, otherwise start from 0
start_epoch = checkpoint.get('epoch', 0)

print("Loading trainset...")
# Load the training dataset
train_dataset = ECGDatasetRandomStart(train_featurevector)  
print("Loading valset...")
# Load the validation dataset
val_dataset = ECGDatasetRandomStart(val_featurevector)

# Create data loaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the loss function
loss = torch.nn.CrossEntropyLoss()
# Initialize the Trainer with the model, loss function, and writer
trainer = Trainer(model, loss, writer)

# Start the epochs from where the checkpoint left off
for epoch in range(start_epoch, 100):
    # Training phase
    trainer.epoch(train_loader, True, epoch)

    # Validation phase
    trainer.epoch(val_loader, False, epoch)