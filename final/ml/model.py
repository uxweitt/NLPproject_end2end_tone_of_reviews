from pathlib import Path
from dataclasses import dataclass
from navec import Navec

import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

class Inference:
    def __init__(self, path_navec, path_model, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.navec = Navec.load(path_navec)
        self.model = model.to(device)
        self.model.load_state_dict(torch.load(path_model))

    def predict_text(self, text):
        words = text.lower().split()
        words = [torch.tensor(self.navec[w]) for w in words if w in self.navec]
        if not words: return {"class_label": "neu", "confidence": 0.0}
            
        _data = torch.stack(words)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(_data.unsqueeze(0).cuda()).squeeze(0)
            probs = F.softmax(logits, dim=0)
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
        y = self.layer(x[:, -1, :])
        return y

@dataclass
class SentimentPrediction:
    label: str
    score: float

def load_model():
    path_navec = BASE_DIR / 'navec_hudlit_v1_12B_500K_300d_100q.tar'
    path_model = BASE_DIR / 'model_gru_semantic.tar'
    model = ReviewModel(300, 3)
    inf = Inference(path_navec, path_model, model)
    
    def predict(text):
        sentiment = inf.predict_text(text)
        return SentimentPrediction(
            label=sentiment['class_label'],
            score=sentiment['confidence'],
        )
    return predict