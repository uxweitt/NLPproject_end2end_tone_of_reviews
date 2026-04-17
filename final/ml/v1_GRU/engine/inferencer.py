from utils.preprocessor import TextPreprocessor
from torch.nn.utils.rnn import pad_sequence

import torch

class Inferencer:
    def __init__(self, model, preprocessor, idx2label, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.idx2label = idx2label
        self.device = device
        
    def _prepare_batch(self, texts):
        sequences = [self.preprocessor.encode_text(text) for text in texts]
        batch = pad_sequence(sequences, batch_first=True)
        return batch.to(self.device)
    
    @torch.inference_mode()
    def predict(self, text):
        batch = self._prepare_batch([text])
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        return {
            "text": text,
            "label": self.idx2label[pred_idx],
            "confidence": float(probs[0, pred_idx].item())
        }
    
    @torch.inference_mode()
    def predict_batch(self, texts):
        batch = self._prepare_batch(texts)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=-1)
        pred_idxs = int(torch.argmax(probs, dim=1).item())
        result = []
        for i, text in enumerate(texts):
            pred_idx = int(pred_idxs[i].item())
            result.append({
                "text": text,
                "label": self.idx2label[pred_idx],
                "confidence": float(probs[i, pred_idx].item())
            })
        return result