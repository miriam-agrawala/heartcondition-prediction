from dataset import ECGDatasetUpdate, DEVICE, ECGDataset200ms, ECGDatasetRandomStart
from import_data import train_featurevector, val_featurevector
from torch.utils.data import Dataset, DataLoader
from lstm import LSTM
from trainloop import Trainer
from lstm_conv import LSTM_Conv
from lstm_2layers_stacked import LSTM_2stacked
from lstm_3layers_stacked import LSTM_3stacked
#from trainloop import val_bacc
from transformer2 import Transformer
import torch

from torch.utils.tensorboard import SummaryWriter  

writer = SummaryWriter()


if __name__ == "__main__":
    # num_samples = 96
    # sequence_length = 1000
    # num_features = 12
    # num_classes = 5
    # dataset = DummyDataset(num_samples, sequence_length, num_features, num_classes, device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading trainset...")
    train_dataset = ECGDatasetRandomStart(train_featurevector)
   
    print("Loading valset...")
    val_dataset = ECGDatasetRandomStart(val_featurevector)


    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # for batch in dataloader:
    #   data_batch, label_batch = batch

    #   # Move the data batch to the same device as the model
    #   data_batch, label_batch = data_batch.to(device), label_batch.to(device)

    net = LSTM_Conv().to(DEVICE)
    net_name = net.__class__.__name__
    #net = Transformer(input_dim=12, output_dim=5, d_model=256, nhead=8, num_layers=2).to(DEVICE)
    loss = torch.nn.CrossEntropyLoss()

    trainer = Trainer(net, loss, writer)
    #trainer.epoch(dataloader, net, True)

    num_epochs = 1000  # number of epochs
    # for epoch in range(num_epochs):
    #     trainer.epoch(dataloader, net, True)
    
    for epoch in range(num_epochs):
        # Training phase
        trainer.epoch(train_loader, True, epoch)

        # Validation phase
        #trainer.epoch(val_loader, False, epoch)
        _, _, val_bacc = trainer.epoch(val_loader, False, epoch)
        
        checkpoint_filename = f"checkpoint_epoch_{epoch}_bacc_{val_bacc:.2f}_net_{net_name}.pth"
        trainer.save_checkpoint(epoch, checkpoint_filename)
        #trainer.save_checkpoint(epoch, 'final_checkpoint.pth')

        #trainer.scheduler.step()

        # net.eval()  # set the model to evaluation mode
        # total_val_loss = 0
        # with torch.no_grad():  # disable gradient computation
        #     for batch, labels in val_loader:
        #         batch, labels = batch.to(DEVICE), labels.to(DEVICE)
        #         outputs = net(batch)  # forward pass
        #         val_loss = loss(outputs, labels)  # compute the validation loss
        #         total_val_loss += val_loss.item()

        # avg_val_loss = total_val_loss / len(val_loader)  # compute the average validation loss