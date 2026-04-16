import torch
import torch.nn as nn

class ReviewModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout_probs=0):
        super(ReviewModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_probs
                            )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru1(x, h0)
        out = self.fc(out[:, -1, :])
        return out