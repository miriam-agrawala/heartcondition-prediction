# import necessary libraries
import torch.nn as nn

class LSTM_3Conv(nn.Module):
    def __init__(self):
        super().__init__()
 
        # Convolutional layers are used to extract features from the input data
        # Each Conv2d layer applies a 2D convolution over an input signal composed of several input planes
        # ReLU (Rectified Linear Unit) activation function is applied after each convolution
        # Batch normalization is commented out, it could be used to normalize the activations of the network if uncommented
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(5,1), stride=(2,1)), nn.ReLU(), #nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(5,1), stride=(2,1)), nn.ReLU(), #nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(5,1), stride=(2,1)), nn.ReLU(), #nn.BatchNorm2d(128),
        )

        # LSTM layer with input size of 128 and hidden size of 256, stacked 2 layers deep
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
        # Dropout layer for regularization, drops out 50% of the input units randomly to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        # Fully connected layer that maps the output of the LSTM layer (256 features) to 5 output features
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
