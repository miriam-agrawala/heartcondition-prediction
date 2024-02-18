# import necessary libraries
import torch.nn as nn

class LSTM_Conv(nn.Module):
    def __init__(self):
        super().__init__()
 
        # Convolutional layers are used to extract features from the input data
        # Each Conv2d layer applies a 2D convolution over an input signal composed of several input planes
        # ReLU (Rectified Linear Unit) activation function is applied after each convolution
        # Batch normalization is commented out, it could be used to normalize the activations of the network if uncommented
        self.conv = nn.Sequential(
           nn.Conv2d(12, 24, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(24),
           nn.Conv2d(24, 48, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(48),
           nn.Conv2d(48, 96, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(96),
           nn.Conv2d(96, 192, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(192),
           nn.Conv2d(192, 384, kernel_size=(10,1), stride=(2,1)), nn.ReLU(), nn.BatchNorm2d(384),
        )    
        
        self.lstm = nn.LSTM(input_size=384, hidden_size=256, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 5)
 
    def forward(self, x):
        # Permute and unsqueeze operations to match the expected input dimensions of the Conv2d layers
        x = x.permute(0,2,1).unsqueeze(3)
        # Pass the input through the convolutional layers
        x = self.conv(x)
        # Squeeze and permute operations to match the expected input dimensions of the LSTM layer      
        x = x.squeeze().permute(0,2,1)
        # Forward pass through LSTM layer
        out, hidden = self.lstm(x)
        # Apply dropout to the LSTM output
        out = self.dropout(out)
        # We apply the fully connected layer to the last output of the LSTM sequence (out[:,-1,:])
        # This will map the output to a tensor of size (batch_size, 5)
        return self.fc(out[:,-1,:])
