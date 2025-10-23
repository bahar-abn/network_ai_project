import torch
import torch.nn as nn
import torch.nn.functional as F

class FCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*(input_dim//2), 32)
        self.fc2 = nn.Linear(32, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)           
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, output_dim=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)
    def forward(self, x):
        x = self.embedding(x)         
        x = self.transformer(x)        
        x = x.mean(dim=1)              
        return self.fc_out(x)
