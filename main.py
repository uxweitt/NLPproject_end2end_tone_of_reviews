import os
import json
import re
import nltk
import emoji
import inflect

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from navec import Navec
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import stopwords

class Inference:
    def __init__(self, path_navec, path_model, model):
        self.navec = Navec.load(path_navec)
        self.model = model
        self.model.load_state_dict(torch.load(path_model))

    def predict(self, text):
        words = text.lower().split()
        words = [torch.tensor(navec[w]) for w in words if w in navec]
        _data = torch.stack(words)
        model.eval()
        with torch.no_grad():
            p = self.model(_data.unsqueeze(0).cuda()).squeeze(0)
            p = torch.argmax(p, dim=1)
        return p
    
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
    
def collate_fn(data):
    tensors, targets = zip(*data)
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    return features, targets

path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)
rd = ReviewDataset('archive/dataset', navec_embedded=navec)

train_d, test_d = data.random_split(rd, [0.8, 0.2])
train_data = data.DataLoader(train_d, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)
test_data = data.DataLoader(test_d, batch_size=8, shuffle=False, collate_fn=collate_fn)

model = ReviewModel(300, 3)
model.cuda()

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
loss_func = nn.CrossEntropyLoss()

epochs = 0
for _e in range(epochs):
    loss_mean = 0
    lm_count = 0
    
    train_tqdm = tqdm(train_data, leave=True)
    model.train()
    for x_train, y_train in train_tqdm:
        predict = model(x_train.cuda()).squeeze(0)
        loss = loss_func(predict, y_train.cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")


inf = Inference(path, 'model_gru_semantic.tar', model)
print(inf.predict("Возьми телефон детка, я знаю ты хочешь позвонить мне сегодня"))
'''st = model.state_dict()
torch.save(st, 'model_gru_semantic.tar')

Q = 0
model.eval()
with torch.no_grad():
    test_tqdm = tqdm(test_data, leave=True)
    for x_test, y_test in test_tqdm:
        predict = model(x_test.cuda()).squeeze(0)
        p = torch.argmax(predict, dim=1)
        y = torch.argmax(y_test.cuda(), dim=1)
        Q += torch.sum(p == y).item()

Q /= len(test_d)
print(Q)'''