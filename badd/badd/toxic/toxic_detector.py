from .file_loader.file_loader import FileLoader
from .toxic.toxic_model import ToxicModel

import torch
import emoji
import re
from nltk.tokenize import WordPunctTokenizer


class ToxicDetector:

    def __init__(self, vocab_path, fasttext_path, model_path, device):
        self._file_loader = FileLoader()
        self._vocab = self._file_loader.load_vocab(vocab_path)
        self._embeddings = self._file_loader.load_fasttext(fasttext_path)
        self._model_path = model_path
        self._device = device
        self._model = ToxicModel(embeddings=self._embeddings,
                                num_classes=1,
                                embedding_dim=300,
                                hidden_dim=254,
                                num_lstm_layers=8)
        self._tokenizer = WordPunctTokenizer().tokenize
        self._max_len = 30
        self.__load_model()

    def __load_model(self):
        if not self._file_loader.check_file(self._model_path):
            raise Exception('badd', f'No file {self._model_path}. Please add it')
        model_state = self._file_loader.load_model(self._model_path)
        self._model.load_state_dict(model_state)
        self._model.to(self._device)
        self._model.eval()

    def __tokenization(self, text):
        result = []
        sentence = self._tokenizer(text)
        for word in sentence:
            detext = emoji.demojize(word)
            detext = re.sub(r"\:[\w]+\:", "", detext)
            if detext:
                result.append(detext)
        return result

    def __binarize(self, pred):
        if pred > 0.5:
            return 1.0
        else:
            return 0.0

    def __get_tensor_indeces(self, text):
        tokenized_text = self.__tokenization(text.lower())
        ids = [self._vocab[token] for token in tokenized_text if token in self._vocab]
        padds = [0] * (self._max_len - len(ids))
        padded_ids = ids + padds
        tensor_ids = torch.tensor(padded_ids)
        return torch.unsqueeze(tensor_ids, 0)

    def predict_text(self, text):
        tensor_ids = self.__get_tensor_indeces(text)
        with torch.no_grad():
            pred = self._model.forward(tensor_ids.to(self._device))
            pred = torch.sigmoid(pred).cpu().numpy()[0][0]
            pred = self.__binarize(pred)
            return pred

    def predict_probability(self, text):
        tensor_ids = self.__get_tensor_indeces(text)
        with torch.no_grad():
            pred = self._model.forward(tensor_ids.to(self._device))
            pred = torch.sigmoid(pred).cpu().numpy()[0][0]
            return pred

    def is_toxic(self, text):
        pred = self.predict_text(text)
        if pred == 1.0:
            return True
        else:
            return False
