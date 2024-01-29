import torch
import torch.nn as nn

class LSTM_3stacked(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=12, hidden_size=128, num_layers=3, batch_first=True)
        #self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128,5)

    def forward(self, x):
        
        out, hidden = self.lstm(x)
        #out = self.dropout(out)
        return self.fc(out[:,-1,:])