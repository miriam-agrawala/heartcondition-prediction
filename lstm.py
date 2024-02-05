import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=12, hidden_size=256, batch_first=True)
        #self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256,5)  

    def forward(self, x):
        #print(x.shape)
        out, hidden = self.lstm(x)
        #exit()
        #out = self.dropout(out)
        return self.fc(out[:,-1,:])