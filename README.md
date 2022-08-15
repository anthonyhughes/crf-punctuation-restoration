# CRF for Punctuation Restoration
Conditional Random Fields model for a punctuation restoration task.

This repo contains the utilities necessary to allow convenient training of a Conditional Random Fields (CRF) model 
for restoration of punctuation to non-punctuated streams of text.

E.g.
`this is my input sentence` becomes `This is my input sentence.`

The model is based on the works of [Lui, M. and Wang, L. (2013), 'Recovering Casing and Punctuation using Conditional Random Fields'](https://aclanthology.org/U13-1020.pdf).

This model treats the task a multi-class token classification task where classification is applied to sequence of words.

The CRF model takes into account the word, POS tag, chunk tags, and NE tags for the current word and two words either side (i.e. 5-gram model)

## Getting started (Local)

### 1. Clone the repository (linux/osx)

```bash
git clone https://github.com/anthonyyhughes/naive-bayes-space-restorer.git
virtualenv env
pip install -r requirements
```

## Getting started (Colab)

### 1. Clone the repository

Recommended method for Google Colab notebooks:

```bash
!git clone https://github.com/anthonyyhughes/naive-bayes-space-restorer.git
!pip install -r requirements
```

## How to use

Example usage for the operations covered below is also included in the example notebook: [crf_punc_restorer_example.ipynb](crf_punc_restorer_example.ipynb).

### Training

Example usage:
```python
from train import train
train('../data/TED_TRAIN.csv')
```

### Read a saved model

```python
from inference import read_trained_model
model = read_trained_model()
```

### Run inference on a single text

```python
from inference import read_trained_model, infer
model = read_trained_model()
infer(model,
          'the anger in me against corruption made me to make a big career change last year becoming a full time '
          'practicing lawyer my experiences over the last 18 months as a lawyer has seeded in me a new entrepreneurial '
          'idea which i believe is indeed worth spreading so i share it with all of you here today though the idea '
          'itself is getting crystallized and im still writing up the business plan of course it helps that fear '
          'of public failure diminishes'
      )
```

####
## References

Lui, M. and Wang, L., ”Recovering Casing and Punctuation using Conditional Random Fields” July, 2018. Available: https://aclanthology.org/U13-1020.pdf.
