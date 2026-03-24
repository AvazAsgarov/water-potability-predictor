import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import INPUT_SIZE, HIDDEN_1, HIDDEN_2, HIDDEN_3, OUTPUT_SIZE

class WaterModel(nn.Module):
    """Feedforward neural network for binary classification of water potability."""
    def __init__(self):
        super(WaterModel, self).__init__()
        
        # Deepened the network slightly compared to the initial notebook
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_1)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.fc3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.out = nn.Linear(HIDDEN_3, OUTPUT_SIZE)
        
        # Dropout added to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        # Sigmoid converts values into probability (0 to 1)
        x = torch.sigmoid(self.out(x))
        return x
