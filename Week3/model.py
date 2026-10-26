import torch
import torch.nn as nn

class ParameterizedLRClassifier(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        # Input size is 4: [x1, x2, mu_s, sigma_s]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)