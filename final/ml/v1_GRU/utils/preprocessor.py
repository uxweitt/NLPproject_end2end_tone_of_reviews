import re
import emoji
import inflect
import torch

from pathlib import Path
from navec import Navec
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, navec_path, emb_dim=300):
        self.navec = Navec.load(str(navec_path))
        self.stop_words = set(stopwords.words("russian"))
        self.inflect_engine = inflect.engine()
        self.emb_dim = emb_dim

    def _emojis_words(self, text):
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = text.replace(":", "").replace("_", " ")
        return text

    def clear_text(self, input_text):
        clean_text = re.sub(r"<[^<]+?>", "", input_text)
        clean_text = re.sub(r"http\S+", "", clean_text)
        clean_text = self._emojis_words(clean_text)
        clean_text = clean_text.lower().replace("\ufeff", "").strip()
        clean_text = re.sub(r"\s+", " ", clean_text)
        clean_text = re.sub(r"[^\w\s]", "", clean_text)
        clean_text = re.sub(r"[^А-яA-z- ]", "", clean_text)

        words = []
        for word in clean_text.split():
            if word.isdigit():
                words.append(self.inflect_engine.number_to_words(word))
            else:
                words.append(word)

        tokens = [w for w in " ".join(words).split() if w not in self.stop_words]
        tokens = [w for w in tokens if w in self.navec]
        return tokens

    def encode_text(self, text):
        tokens = self.clear_text(text)
        if not tokens:
            return torch.zeros(1, self.emb_dim, dtype=torch.float32)

        vectors = [torch.tensor(self.navec[token], dtype=torch.float32) for token in tokens]
        return torch.vstack(vectors)