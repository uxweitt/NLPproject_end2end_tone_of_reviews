from navec import Navec
from nltk.corpus import stopwords

import os
import json
import re
import nltk
import emoji
import inflect
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

class Inference:
    def __init__(self, model, navec, path_model=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.navec = navec
        self.model = model.to(device)
        if path_model:
            self.model.load_state_dict(torch.load(path_model))

    def predict_text(self, text):
        words = text.lower().split()
        words = [torch.tensor(self.navec[w]) for w in words if w in self.navec]
        if not words: return {"class_label": "neu", "confidence": 0.0}
        
        _data = torch.stack(words)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(_data.unsqueeze(0).cuda()).squeeze(0)
            probs = F.softmax(logits, dim=0).squeeze()
            pred_cl = torch.argmax(probs).item()
            confid = probs[pred_cl].item()
            dct = {
                0: "neg",
                1: "neu",
                2: "pos",
            }
        return {
            'class_label': dct[pred_cl],
            'confidence': round(confid, 3)
        }
        
class ReviewModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 32
        self.in_features = in_features
        self.out_features = out_features
        
        self.gru1 = nn.GRU(self.in_features, self.hidden_size, batch_first=True)
        self.layer = nn.Linear(self.hidden_size, self.out_features)
        
    def forward(self, x):
        x, h = self.gru1(x)
        y = self.layer(h)
        return y
    
class ReviewDataset(data.Dataset):
    def __init__(self, path, navec_embedded):
        
        nltk.download('stopwords')
        
        self.navec_embedded = navec_embedded
        self.path = path
        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format = json.load(fp)
            
        self.length = 0
        self.files = [] # (path2txt, target)
        self.targets = torch.eye(3)
        
        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)[:15000]
            self.length += len(list_files)
            self.files.extend(
                map(
                lambda _x: 
                (os.path.join(path, _x), _target), list_files
                ))
            
    def __getitem__(self, item):
        path_file, target = self.files[item]
        targ = self.targets[target]
        with open(path_file, 'r', encoding='utf-8') as f:
            txt = f.read()
            txt_list = self._clear_text(txt)
        
        words_embedded = []
        for word in txt_list:
            t = torch.tensor(self.navec_embedded[word], dtype=torch.float32)
            words_embedded.append(t)
            
        return torch.vstack(words_embedded), targ
        
    def _clear_text(self, input_text):
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
        _words = [w for w in _words if w in self.navec_embedded]
        return _words
        
    def __len__(self):
        return self.length
    
    def collate_fn(self, data):
        tensors, targets = zip(*data)
        features = pad_sequence(tensors, batch_first=True)
        targets = torch.stack(targets)
        return features, targets
    
class ProcessModel:
    def __init__(self, ds, model, epochs, collate_fn, path_navec, path_model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ds = ds
        self.model = model.to(self.device)
        self.path_model = path_model
        self.optimizator = optim.Adam(params=self.model.parameters(), lr=0.05, weight_decay=0.001)
        self.loss_func = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.path_navec = path_navec
        self.navec = Navec.load(self.path_navec)
        self.collate_fn = collate_fn
        
    def _ds_init(self, ptrain, ptest):
        self.d_train, self.d_test = data.random_split(self.ds, [ptrain, ptest])
        self.train_data = data.DataLoader(self.d_train, batch_size=16, shuffle=True, drop_last=True, collate_fn=self.collate_fn)
        self.test_data = data.DataLoader(self.d_test, batch_size=8, shuffle=False, collate_fn=self.collate_fn)
        
    def _train_loop(self):
        for _e in range(self.epochs):
            loss_mean = 0
            lm_count = 0
            train_tqdm = tqdm(self.train_data, leave=True)
            self.model.train()
            for x_train, y_train in train_tqdm:
                predict = self.model(x_train.cuda()).squeeze(0)
                loss = self.loss_func(predict, y_train.cuda())
                self.optimizator.zero_grad()
                loss.backward()
                self.optimizator.step()
                lm_count += 1
                loss_mean = 1/lm_count*loss.item() + (1-1/lm_count)*loss_mean
                train_tqdm.set_description(f"Epoch [{_e+1}/{self.epochs}], loss_mean={loss_mean:.3f}")
    
    def _test_loop(self):
        Q = 0
        self.model.eval()
        with torch.no_grad():
            test_tqdm = tqdm(self.test_data, leave=True)
            for x_test, y_test in test_tqdm:
                predict = self.model(x_test.cuda()).squeeze(0)
                p = torch.argmax(predict, dim=1)
                y = torch.argmax(y_test.cuda(), dim=1)
                Q += torch.sum(p == y).item()
        self.quality = Q / len(self.d_test)
        
    def _inference(self, text):
        inf = Inference(self.model, self.navec)
        return inf.predict_text(text)