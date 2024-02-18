# import necessary libraries
import torch.nn as nn

class LSTM_3stacked(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize a 3-layer LSTM with input size of 12 and hidden size of 128
        # batch_first=True makes the input/output tensors to be provided as (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=12, hidden_size=128, num_layers=3, batch_first=True)

        # Dropout layer is commented out, it could be used for regularization if uncommented
        # self.dropout = nn.Dropout(0.5)

        # Fully connected layer that maps the output of the LSTM layer (128 features) to 5 output features
        self.fc = nn.Linear(128,5)

    def forward(self, x):
        # Forward pass through LSTM layer
        # out: tensor of shape (batch_size, seq_length, hidden_size) containing the output features from the last layer of the LSTM
        # (hidden: (h_n, c_n) for all timesteps)
        out, hidden = self.lstm(x)
        # Dropout layer is commented out, it could be used for regularization if uncommented
        # out = self.dropout(out)

        # We apply the fully connected layer to the last output of the LSTM sequence (out[:,-1,:])
        # This will map the output to a tensor of size (batch_size, 5)
        return self.fc(out[:,-1,:])