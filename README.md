# BadD(etector)

## Description (How it can help)

BadD(etector) was created for detecting bad things in user-generated content in Russian.
Now this library supports obscene words detection, advertising detection and toxicity detection. 
All the magic done by neural networks.

## Requirements
1. Python 3.7+
1. PyTorch 1.8.1
1. Gensim 3.8.1
1. NLTK 3.2.5
1. pymorphy2 0.9.1
1. emoji 1.2.0

## How to install

**locally (dev mode)**
```shell
python3 -m pip install -e <path-to-lib>
```

**from github**
```shell
pip install git+https://github.com/wksmirnowa/badd.git@master
```

**from pip**
```shell
pip install badd
```

## Usage

Download files and models for:

* [ObsceneDetector](https://drive.google.com/file/d/1Q2rVfRHFKMV97Fa7n-Ll0NPcZoZeD09m/view?usp=sharing)
* [AdDetector](https://drive.google.com/file/d/1EvbLwT7r66wAI29wDBQrMFSc00qXGYXy/view?usp=sharing)
* [ToxicDetector](https://drive.google.com/file/d/1IHHdnCycu8OKrHMqVp-xXNJmNZyErMll/view?usp=sharing)


### Obscene words detection

Import the ObsceneDetector class

```python3
import torch
from badd import ObsceneDetector
```

Set pathes to files and device

```python3
# path to vocab
vocab_path = "obscene_vocab.json"
# path to embeddings
fasttext_path = "obscene_embeddings.pickle"
# path to model 
model_path = "obscene_model_cpu.pth"
# set device
device = torch.device('cpu')
```

Use ObsceneDetector

```python3
obscene_detector = ObsceneDetector(vocab_path, fasttext_path, model_path, device)
```

Predict every word in text

```python3
obscene_detector.predict_text(text)
```

Predict probability for every word in text

```python3
obscene_detector.predict_probability(text)
```

Check whether any obscene word is in text

```python3
obscene_detector.obscene_in_text(text)
```

#### Attributes

* ```obscene_detector.obscene_words``` list of found obscene words. Available after one of the methods 
(```predict_probability```, ```predict_text```, ```obscene_in_text```) was runned.

### Ad detection

Import the AdDetector class

```python3
import torch
from badd import AdDetector
```

Set pathes to files and device

```python3
# path to vocab
vocab_path = "ad_vocab.json"
# path to embeddings
fasttext_path = "ad_embeddings.pickle"
# path to model 
model_path = "ad_model_cpu.pth"
# set device
device = torch.device('cpu')
```

Use AdDetector

```python3
ad_detector = AdDetector(vocab_path, fasttext_path, model_path, device)
```

Predict text

```python3
ad_detector.predict_text(text)
```

Predict probability for text

```python3
ad_detector.predict_probability(text)
```

Check whether a text is ad

```python3
ad_detector.is_ad(text)
```

### Toxic texts detection

Import the ToxicDetector class

```python3
import torch
from badd import ToxicDetector
```

Set pathes to files and device

```python3
# path to vocab
vocab_path = "toxic_vocab.json"
# path to embeddings
fasttext_path = "toxic_embeddings.pickle"
# path to model 
model_path = "toxic_model_cpu.pth"
# set device
device = torch.device('cpu')
```

Use AdDetector

```python3
toxic_detector = ToxicDetector(vocab_path, fasttext_path, model_path, device)
```

Predict text

```python3
toxic_detector.predict_text(text)
```

Predict probability for text

```python3
toxic_detector.predict_probability(text)
```

Check whether a text is toxic

```python3
toxic_detector.is_toxic(text)
```
