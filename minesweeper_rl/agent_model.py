# agent_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentModel(nn.Module):
    def __init__(self, rows, cols):
            super().__init__()
            self.rows = rows
            self.cols = cols
            
            # 5 input channels
            self.conv_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(5, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                )
            ])
            
            self.attention = nn.Sequential(
                nn.Conv2d(128, 32, 1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            )
            
            with torch.no_grad():
                dummy = torch.zeros(1, 5, rows, cols)
                cnn_out = self._cnn_forward(dummy)
                self.cnn_output_size = cnn_out.view(1, -1).shape[1]
            
            self.fc = nn.Sequential(
                nn.Linear(self.cnn_output_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(256, rows * cols)
            )
        
    def _cnn_forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        attention_weights = self.attention(x)
        x = x * attention_weights
        return x
        
    def forward(self, x):
        features = self._cnn_forward(x)
        features = features.view(features.size(0), -1)
        q_values = self.fc(features)
        return q_values