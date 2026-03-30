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
    path_navec = BASE_DIR / 'navec_hudlit_v1_12B_500K_300d_100q.tar'
    path_model = BASE_DIR / 'model_gru_semantic.tar'
    model = model_space.ReviewModel(300, 3)
    inf = model_space.Inference(path_navec, path_model, model)
    
    def predict(text):
        sentiment = inf.predict_text(text)
        return SentimentPrediction(
            label=sentiment['class_label'],
            score=sentiment['confidence'],
        )
    return predict