# import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader

# Import Tensorboard writer for logging
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from dataset import DEVICE, ECGDataset200, ECGDatasetUpdate, ECGDatasetRandomStart
from import_data import train_featurevector, val_featurevector
from trainloop import Trainer
from lstm import LSTM
from lstm_2layers_stacked import LSTM_2stacked
from lstm_3layers_stacked import LSTM_3stacked
from lstm_conv import LSTM_Conv

# Initialize a SummaryWriter for TensorBoard
writer = SummaryWriter()

# Main execution
if __name__ == "__main__":

    # Load training dataset
    print("Loading trainset...")
    train_dataset = ECGDatasetRandomStart(train_featurevector)
    # Load validation dataset
    print("Loading valset...")
    val_dataset = ECGDatasetRandomStart(val_featurevector)

    # Create data loaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Initialize the model and move it to the device (CPU or GPU)
    net = LSTM_Conv().to(DEVICE)
    # net_name is used for naming the checkpoint files
    net_name = net.__class__.__name__

    # Define the loss function
    loss = torch.nn.CrossEntropyLoss()

    # Initialize the Trainer with the model, loss function, and writer
    trainer = Trainer(net, loss, writer)
    
    num_epochs = 10  # number of epochs
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        trainer.epoch(train_loader, True, epoch)

        # Validation phase
        _, _, val_bacc = trainer.epoch(val_loader, False, epoch)

        # Save a checkpoint after each epoch
        checkpoint_filename = f"checkpoint_epoch_{epoch}_bacc_{val_bacc:.2f}_net_{net_name}.pth"
        trainer.save_checkpoint(epoch, checkpoint_filename)

        # Uncomment the following line if you want to use a learning rate scheduler
        #trainer.scheduler.step()
