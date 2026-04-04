from pathlib import Path
from dataclasses import dataclass

from final.ml import model_space

import yaml

BASE_DIR = Path(__file__).resolve().parent
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

@dataclass
class SentimentPrediction:
    label: str
    score: float

def load_model():
    path_navec = BASE_DIR / config['navec']
    path_model = BASE_DIR / config['model']
    model = model_space.ReviewModel(300, 3)
    inf = model_space.Inference(model, navec=None, path_model=path_model, path_navec=path_navec)
    
    def predict(text):
        sentiment = inf.predict_text(text)
        return SentimentPrediction(
            label=sentiment['class_label'],
            score=sentiment['confidence'],
        )
    return predict