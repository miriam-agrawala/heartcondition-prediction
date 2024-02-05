import torch
import torch.nn as nn

class LSTM_Conv(nn.Module):
    def __init__(self):
        super().__init__()
 
        #self.conv = nn.Sequential(
        #    nn.Conv2d(12, 24, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(24),
        #    nn.Conv2d(24, 48, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(48),
         #   nn.Conv2d(48, 96, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(96),
        #    nn.Conv2d(96, 192, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(192),
        #    nn.Conv2d(192, 384, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(384),
        # 
        #     )
      
        
        #self.lstm = nn.LSTM(input_size=384, hidden_size=256, batch_first=True)
        #self.dropout = nn.Dropout(0.5)
        #self.fc = nn.Linear(256, 5)

        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(5,1), stride=(2,1)), nn.ReLU(), #nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(5,1), stride=(2,1)), nn.ReLU(), #nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(5,1), stride=(2,1)), nn.ReLU(), #nn.BatchNorm2d(128),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 5)

        #self.conv = nn.Sequential(
        #    nn.Conv2d(12, 32, kernel_size=(5,1), stride=(2,1)), nn.ReLU(),#nn.BatchNorm2d(32),
        #    nn.Conv2d(32, 64, kernel_size=(5,1), stride=(2,1)), nn.ReLU(), #nn.BatchNorm2d(64),
        #    nn.Conv2d(64, 128, kernel_size=(5,1), stride=(2,1)), nn.ReLU(), #nn.BatchNorm2d(128),
        #)

        #self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        #self.dropout = nn.Dropout(0.5)
        #self.fc = nn.Linear(128, 5)

 
    def forward(self, x):
        BS = x.shape[0]

        x = x.permute(0,2,1).unsqueeze(3)
        x = self.conv(x)        
        x = x.squeeze().permute(0,2,1)
        # x = x.view(-1, 12, 1000, 1)
        # x = self.conv(x)
        # #x = x.view(BS,23,384)
        # x = x.view(BS,118,96)

        out, hidden = self.lstm(x)
        out = self.dropout(out)
         
        return self.fc(out[:,-1,:])
 
 
# x = torch.rand((64, 1000, 12))
# net = Net()
# y = net(x)
# print(x.shape)
# print(y.shape)