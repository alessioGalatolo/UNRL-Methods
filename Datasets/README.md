# Datasets Module
The module ```datasets.py``` wants to be an helper to access the various datasets.
Given the Dataset name there is a function to return the respective graph.

Usage:
```python
    from Datasets.datasets import Datasets, get_graph
    graph = get_graph(Datasets.PubMed)
```

The available datasets are:
1.    Twitter
1.    BlogCatalog
1.    YouTube
1.    Flickr
1.    CitHepPh
1.    Reddit - the retrival of this dataset has not been implemented due to its size
1.    Cora
1.    Epinions
1.    Google
1.    Wiki
1.    Pubmed
1.    LastFm

# Sources - Original datasets
1. blogcatalog, flickr - http://leitang.net/social_dimension.html
1. youtube - original required permission from mp: http://socialnetworks.mpi-sws.org/datasets.html so the dataset from http://leitang.net/social_dimension.html was used instead
1. reddit - http://snap.stanford.edu/graphsage/
1. Twitter - original is not available anymore: http://www.public.asu.edu/~mdechoud/temp/released-data/ so dataset from https://snap.stanford.edu/data/ego-Twitter.html was used instead
1. Epinions - original is not available anymore: http://www.bibserv.org/ so dataset from https://snap.stanford.edu/data/soc-Epinions1.html was used instead
1. DBLP-Ci - Not available: https://dblp.uni-trier.de/dblpbr/index.html
1. CoCit - Not available: http://datamarket.azure.com/dataset/mrc/microsoftacademic
1. Cora - https://graphsandnetworks.com/the-cora-dataset/
1. Pubmed - https://web.archive.org/web/20150918182432/http://linqs.umiacs.umd.edu/projects//projects/lbc/Pubmed-Diabetes.tgz
1. DBLP-Au - https://lfs.aminer.cn/lab-datasets/citation/DBLP_citation_2014_May.zip

# Soruces - New Datasets
1. LastFM Asia Social Network - https://snap.stanford.edu/data/feather-lastfm-social.html
1. Google web graph - https://snap.stanford.edu/data/web-Google.html