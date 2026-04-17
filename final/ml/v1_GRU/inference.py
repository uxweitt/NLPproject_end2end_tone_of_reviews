from engine.inferencer import Inferencer
from utils.preprocessor import TextPreprocessor
from models.model import ReviewModel

from pathlib import Path

import torch

def main():
    ex_text = """
        Работай чатджипити родненький.
    """
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = "models"
    MODEL_NAME = "best_loss_val_0.139.pt"
    model = ReviewModel(input_size=300,
                        hidden_size=64,
                        num_classes=3,
                        num_layers=4,
                        dropout_probs=1e-3)
    model.load_state_dict(torch.load(BASE_DIR/MODEL_DIR/MODEL_NAME,
                                    weights_only=True))
    navec_path = Path(__file__).resolve().parent / "data_set" / "navec_hudlit_v1_12B_500K_300d_100q.tar"
    idx2label = {
        0: "neg",
        1: "neu",
        2: "pos",
    }
    params = {
        "model": model,
        "preprocessor": TextPreprocessor(navec_path),
        "idx2label": idx2label,
        "device": 'cpu',
    }
    inf = Inferencer(**params)
    print(inf.predict(ex_text))
    
if __name__ == "__main__":
    main()