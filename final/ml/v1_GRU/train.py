from  engine.trainer import Trainer
from data_set.data_set import ReviewDataset
from models.model import ReviewModel

import torch.optim
import torch.nn as nn

def main():
    params = {
    'dataset': ReviewDataset("archive\dataset", 3),
    'net': ReviewModel(input_size=300, hidden_size=64, num_classes=3, num_layers=4, dropout_probs=1e-3),
    'epoch_amount': 5, 
    'learning_rate': 1e-3,
    'early_stopping': 25,
    'loss_f': nn.CrossEntropyLoss(),
    'optim': torch.optim.Adam,
    'device': 'cuda',
    'save_best': True
    }
    clf = Trainer(**params)
    clf.fit()
    
if __name__ == "__main__":
    main()