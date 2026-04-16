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

BASE_DIR = Path(__file__).resolve().parent
NAVEC_DIR = "navec_hudlit_v1_12B_500K_300d_100q.tar"
print(BASE_DIR / NAVEC_DIR)

class ReviewDataset(data.Dataset):
    def __init__(self, data_dir, num_classes):
        nltk.download('stopwords')
        self.data_dir = data_dir            
        self.length = 0
        self.files = [] # (path2txt, target)
        self.targets = torch.eye(num_classes)
        self.navec = Navec.load(BASE_DIR / NAVEC_DIR)
        
        with open(os.path.join(self.data_dir, "format.json"), "r") as fp:
            self.format = json.load(fp)
            
        for _dir_class, _target in self.format.items():
            path = os.path.join(self.data_dir, _dir_class)
            list_files = os.listdir(path)[:150] # Можно взять [:15000] для баланса всех классов
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
            txt_list = self.clear_text(txt)
        
        words_embedded = []
        for word in txt_list:
            t = torch.tensor(self.navec[word], dtype=torch.float32)
            words_embedded.append(t)
            
        return torch.vstack(words_embedded), emb_target
    
    def __len__(self):
        return self.length
    
    def clear_text(self, input_text):
        def emojis_words(text):
            clean_text = emoji.demojize(text, delimiters=(" ", " "))
            clean_text = clean_text.replace(":", "").replace("_", " ")
            return clean_text
        clean_text = re.sub('<[^<]+?>', '', input_text) # Удаление HTML тегов
        clean_text = re.sub(r'http\S+', '', clean_text) # Удаление URL и ссылок
        clean_text = emojis_words(clean_text)           # Обрабатываем эмодзи
        clean_text = clean_text.lower().replace('\ufeff', '').strip()
        clean_text = re.sub('\s+', ' ', clean_text)
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        clean_text = re.sub(r'[^А-яA-z- ]', '', clean_text)
        
        temp = inflect.engine()
        words = []
        for word in clean_text.split():
            if word.isdigit():
                words.append(temp.number_to_words(word))
            else:
                words.append(word)

        clean_text = ' '.join(words)
        stop_words = set(stopwords.words('russian'))
        _words = clean_text.split()
        _words = [token for token in _words if token not in stop_words]
        _words = [w for w in _words if w in self.navec]
        return _words
    
def collate_fn(data):
        tensors, targets = zip(*data)
        features = pad_sequence(tensors, batch_first=True)
        targets = torch.stack(targets)
        return features, targets