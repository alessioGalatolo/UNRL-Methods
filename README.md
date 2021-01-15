# UNRL Methods
This repository will contain the following methods for network embedding:
1. DeepWalk
2. Node2Vec
3. Line-1
4. NetMF
5. GraphSage

Each method is contained in its own package (random_walk_based contains 2 methods). This is to facilitate the debugging of individual methods. There is a package with all the datasets that are used and some usefull functions.

# Usage
Basic usage is:
```bash
pip install -r requirements.txt
python link_prediction.py
python node_classification.py
```
Usage with a virtual environment (linux):
```bash
pip install virtualenv
python -m venv env
source ./env
pip install -r requirements.txt
python link_prediction.py
python node_classification.py
```

A possible script to test the methods is:
```python
from Datasets.datasets import Datasets, get_graph
from random_walk_based.deepwalk import deepwalk

dataset = Datasets.Cora
embedding = deepwalk(get_graph(dataset))
```

# TODOs
Embedding methods:
1. DeepWalk - 100%
2. Node2Vec - 100%
3. Line-1 - 100%
4. NetMF - 100%
5. GraphSage - I want to cry

Search of original datasets 90%
Missing: DBLP-Au, CoCit

Search of new datasets
1. https://snap.stanford.edu/data/wikipedia-article-networks.html
2. https://snap.stanford.edu/data/ego-Facebook.html

Implementation of tasks:
1. Link prediction - 100%
2. Node classification - 100%
