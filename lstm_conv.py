import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.conv = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=(10,1), stride=(2,1)), nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=(10,1), stride=(2,1)), nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=(10,1), stride=(2,1)), nn.ReLU(),
            )
       
        self.lstm = nn.LSTM(input_size=96, hidden_size=128, batch_first=True)
 
        self.fc = nn.Linear(128, 5)
 
    def forward(self, x):
        x = x.view(-1, 12, 1000, 1)
        x = self.conv(x).view(-1,118,96)
        out, hidden = self.lstm(x)
        return self.fc(out[:,-1,:])
 
 
x = torch.rand((64, 1000, 12))
net = Net()
y = net(x)
print(x.shape)
print(y.shape)