# BadD(etector)

## Description (How it can help)

## Requirements
1. Python 3.7+
1.
1.

## How to install

**locally (dev mode)**
```shell
python3 -m pip install -e <path-to-lib>
```

**from github**
```shell
pip3 install git+https://github.com/wksmirnowa/badd.git@master
```

**from pip**
```shell
pip3 install badd
```

## Usage
### Obscene words detection

Import the ObsceneDetector class

```python3
from badd import ObsceneDetector
```

Set pathes to files and device

```python3
# path to vocab
vocab_path = "obscene_vocab_cpu.json"
# path to embeddings
fasttext_path = "obscene_embeddings.pickle"
# path to model 
model_path = "obscene_model_best_cpu.pth"
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
from badd import AdDetector
```

Set pathes to files and device

```python3
# path to vocab
vocab_path = "ad_vocab_cpu.json"
# path to embeddings
fasttext_path = "ad_embeddings.pickle"
# path to model 
model_path = "ad_model_best_cpu.pth"
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
from badd import ToxicDetector
```

Set pathes to files and device

```python3
# path to vocab
vocab_path = "toxic_vocab_cpu.json"
# path to embeddings
fasttext_path = "toxic_embeddings.pickle"
# path to model 
model_path = "toxic_model_best_cpu.pth"
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
