from .file_loader import FileLoader
from .obscene_model import ObsceneModel

import torch
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer


class ObsceneDetector:

    def __init__(self, vocab_path, fasttext_path, model_path, device):
        self._file_loader = FileLoader()
        self._vocab = self._file_loader.load_vocab(vocab_path)
        self._embeddings = self._file_loader.load_fasttext(fasttext_path)
        self._model_path = model_path
        self._device = device
        self._model = ObsceneModel(embeddings=self._embeddings,
                                  num_classes=1,
                                  linear_size_1=254,
                                  linear_size_2=128)
        self._tokenizer = WordPunctTokenizer().tokenize
        self.morph = MorphAnalyzer()
        self.obscene_words = []
        self.__load_model()

    def __load_model(self):
        if not self._file_loader.check_file(self._model_path):
            raise Exception('badd', f'No file {self._model_path}. Please add it')
        model_state = self._file_loader.load_model(self._model_path)
        self._model.load_state_dict(model_state)
        self._model.to(self._device)
        self._model.eval()

    def __binarize(self, pred):
        if pred > 0.5:
            return 1.0
        else:
            return 0.0

    def __get_tensor_index(self, word):
        word = self.morph.parse(word)[0].normal_form
        index = self._vocab[word] if word in self._vocab else 0
        tensor_index = torch.tensor(index)
        return torch.unsqueeze(tensor_index, 0)

    def __predict_word(self, word):
        tensor_index = self.__get_tensor_index(word)
        with torch.no_grad():
            pred = self._model.forward(tensor_index.to(self._device))
            pred = torch.sigmoid(pred).cpu().numpy()[0][0]
            pred = self.__binarize(pred)
            return pred

    def __predict_probability_word(self, word):
        tensor_index = self.__get_tensor_index(word)
        with torch.no_grad():
            pred = self._model.forward(tensor_index.to(self._device))
            pred = torch.sigmoid(pred).cpu().numpy()[0][0]
            return pred

    def predict_text(self, text):
        self.obscene_words = []
        tokenized_text = self._tokenizer(text.lower())
        preds = [self.__predict_word(token) for token in tokenized_text]
        words = {t: p for t, p in zip(tokenized_text, preds)}
        self.obscene_words = [word for word in words.keys() if words[word] == 1.0]
        return preds

    def predict_probability(self, text):
        self.obscene_words = []
        tokenized_text = self._tokenizer(text.lower())
        preds = [self.__predict_probability_word(token) for token in tokenized_text]
        words = {t: p for t, p in zip(tokenized_text, preds)}
        self.obscene_words = [word for word in words.keys() if words[word] > 0.5]
        return preds

    def obscene_in_text(self, text):
        preds = self.predict_text(text)
        if 1.0 in preds:
            return True
        else:
            return False
