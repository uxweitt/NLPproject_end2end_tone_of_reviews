import os
import json
import re
import emoji
import inflect
import nltk

import torch
import torch.utils.data as data

from navec import Navec
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from utils.preprocessor import TextPreprocessor

BASE_DIR = Path(__file__).resolve().parent
NAVEC_DIR = "navec_hudlit_v1_12B_500K_300d_100q.tar"
print(BASE_DIR / NAVEC_DIR)

class ReviewDataset(data.Dataset):
    def __init__(self, data_dir, num_classes, preprocessor=TextPreprocessor):
        self.data_dir = data_dir            
        self.length = 0
        self.files = [] # (path2txt, target)
        self.targets = torch.eye(num_classes)
        navec_path = BASE_DIR / NAVEC_DIR
        self.preprocessor = preprocessor(navec_path)
        
        with open(os.path.join(self.data_dir, "format.json"), "r") as fp:
            self.format = json.load(fp)
            
        for _dir_class, _target in self.format.items():
            path = os.path.join(self.data_dir, _dir_class)
            list_files = os.listdir(path) # Можно взять [:15000] для баланса всех классов
            self.length += len(list_files)
            self.files.extend(
                        map(lambda _x: (os.path.join(path, _x), _target), 
                        list_files
                        ))
            
    def __getitem__(self, item):
        path_file, target = self.files[item]
        emb_target = self.targets[target]
        with open(path_file, 'r', encoding='utf-8') as f:
            txt = f.read()
            
        vectors = self.preprocessor.encode_text(txt)    
        return vectors, emb_target
    
    def __len__(self):
        return self.length
    
    
def collate_fn(data):
        tensors, targets = zip(*data)
        features = pad_sequence(tensors, batch_first=True)
        targets = torch.stack(targets)
        return features, targets