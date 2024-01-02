from dataset import ECGDataset, DEVICE
from torch.utils.data import Dataset, DataLoader
from lstm2 import LSTM2
from trainloop import Trainer
from lstm_conv import Net
import torch


if __name__ == "__main__":
    # num_samples = 96
    # sequence_length = 1000
    # num_features = 12
    # num_classes = 5
    # dataset = DummyDataset(num_samples, sequence_length, num_features, num_classes, device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ECGDataset(seqlen=4)


    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for batch in dataloader:
    #   data_batch, label_batch = batch

    #   # Move the data batch to the same device as the model
    #   data_batch, label_batch = data_batch.to(device), label_batch.to(device)
    net = Net().to(DEVICE)
    loss = torch.nn.CrossEntropyLoss()

    trainer = Trainer(net, loss)
    #trainer.epoch(dataloader, net, True)

    num_epochs = 100  # number of epochs
    for epoch in range(num_epochs):
        trainer.epoch(dataloader, net, True)