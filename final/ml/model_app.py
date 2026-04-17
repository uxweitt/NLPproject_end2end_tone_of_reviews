from pathlib import Path
from dataclasses import dataclass

from final.ml.v1_GRU.models.model import ReviewModel
from final.ml.v1_GRU.engine.inferencer import Inferencer
from final.ml.v1_GRU.utils.preprocessor import TextPreprocessor

import yaml
import torch

BASE_DIR = Path(__file__).resolve().parent
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

@dataclass
class SentimentPrediction:
    text: str
    label: str
    confidence: float

def load_model():
    path_navec = BASE_DIR / config['navec']
    path_model = BASE_DIR / config['model']
    model = ReviewModel(input_size=300,
                        hidden_size=64,
                        num_classes=3,
                        num_layers=4,
                        dropout_probs=1e-3)
    model.load_state_dict(torch.load(BASE_DIR / path_model))
    idx2label = {
        0: "neg",
        1: "neu",
        2: "pos",
    }
    params_inf = {
        "model": model,
        "preprocessor": TextPreprocessor(path_navec),
        "idx2label": idx2label,
        "device": 'cpu',
    }
    inf = Inferencer(**params_inf)
    
    def model(text):
        sentiment = inf.predict(text)
        return SentimentPrediction(
            text=sentiment['text'],
            label=sentiment['label'],
            confidence=sentiment['confidence'],
        )
    return model