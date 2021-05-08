import os
import torch
import json
import pickle
from collections import OrderedDict

class FileLoader:

    def __init__(self):
        pass

    def check_file(self, file_name: str) -> bool:
        return os.path.isfile(file_name)

    def load_model(self, file_name: str) -> str:
        print("Load model")
        model_state = torch.load(file_name)
        return model_state

    def load_fasttext(self, file_name: str) -> str:
        print("Load embeddings")
        with open(file_name, 'rb') as f:
            embeddings = pickle.load(f)
        return torch.tensor(embeddings).float()

    def load_vocab(self, file_name: str) -> str:
        print("Load vocab")
        with open(file_name, "r") as f:
            vocab_file = f.read()
        vocab = OrderedDict(json.loads(vocab_file))
        return vocab
