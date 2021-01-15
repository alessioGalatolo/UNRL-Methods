"""Made by Alessio Galatolo
This module wants to be an helper to access the various datasets.
Given the Dataset name there is a function to return the respective graph.

Usage:
    from Datasets.datasets import Datasets, get_graph
    graph = get_graph(Datasets.PubMed)

The available datasets are shown below
"""

from enum import Enum, auto
from networkx import Graph, DiGraph, is_weighted

#list of available datasets
class Datasets(Enum):
    Twitter = auto()
    BlogCatalog = auto()
    YouTube = auto()
    Flickr = auto()
    CitHepPh = auto()
    Reddit = auto()
    Cora = auto()
    Epinions = auto()
    Google = auto()
    Wiki = auto()
    Pubmed = auto()
    LastFm = auto()

class Formats(Enum):
    mat = auto()
    txt = auto() #all graphs with this format are directed
    csv = auto()
    
#given the dataset name will return the filename
dataset2filename = {Datasets.Twitter: "Datasets/twitter_combined.txt",
                    Datasets.BlogCatalog: "Datasets/blogcatalog/",
                    Datasets.YouTube: "Datasets/youtube.mat",
                    Datasets.Flickr: "Datasets/flickr.mat",
                    Datasets.CitHepPh: "Datasets/Cit-HepPh.txt",
                    Datasets.Reddit: "Datasets/reddit/", #todo
                    Datasets.Cora: "Datasets/cora/",
                    Datasets.Epinions: "Datasets/soc-Epinions1.txt",
                    Datasets.Google: "Datasets/web-Google.txt",
                    Datasets.Wiki: "Datasets/Wiki/",
                    Datasets.Pubmed: "Datasets/Pubmed-Diabetes/",
                    Datasets.LastFm: "Datasets/lastfm_asia/"
                    }

#given the dataset name will return the format (.mat, .csv, list of edges or others)
dataset2format = {Datasets.Twitter: Formats.txt,
                    Datasets.BlogCatalog: Formats.csv,
                    Datasets.YouTube: Formats.mat,
                    Datasets.Flickr: Formats.mat,
                    Datasets.CitHepPh: Formats.txt,
                    Datasets.Reddit: "Datasets/reddit/", #todo
                    Datasets.Cora: Formats.csv,
                    Datasets.Epinions: Formats.txt,
                    Datasets.Google: Formats.txt,
                    Datasets.Wiki: Formats.csv,
                    Datasets.Pubmed: Formats.csv,
                    Datasets.LastFm: Formats.csv
                }

dataset2directionality = {Datasets.Twitter: DiGraph(),
                            Datasets.BlogCatalog: Graph(),
                            Datasets.YouTube: Graph(),
                            Datasets.Flickr: Graph(),
                            Datasets.CitHepPh: DiGraph(),
                            Datasets.Reddit: Graph(),
                            Datasets.Cora: DiGraph(),
                            Datasets.Epinions: DiGraph(),
                            Datasets.Google: DiGraph(),
                            Datasets.Wiki: Graph(),
                            Datasets.Pubmed: DiGraph(),
                            Datasets.LastFm: Graph()
                        }

def get_graph(dataset: Datasets):
    #datasets in mat format
    filename = dataset2filename[dataset]
    format = dataset2format[dataset]
    graph = dataset2directionality[dataset] #the graph to return 
    
    if format is Formats.mat:
        from scipy.io import loadmat
        data = loadmat(filename) 
        #data['network'] contains the graph, while data['group'] the labels
        #both are sparse matrix
        sparse_graph = data['network']
        for u, v, w in zip(*sparse_graph.nonzero(), sparse_graph.data):
            graph.add_edge(u, v, weight=w)

    elif format is Formats.txt:
        from networkx import read_edgelist
        graph = read_edgelist(filename, create_using=graph, nodetype=int, data=(('weight',float),))

    elif format is Formats.csv:
        from pandas import read_csv
        data = read_csv(filename + "edges.csv") 
        
        for u, v in data.values:
            graph.add_edge(u, v, weight=1)

    #check if graph has weights - if not add uniform
    if not is_weighted(graph):
        from networkx import set_edge_attributes
        set_edge_attributes(graph, values = 1, name = 'weight')

    return graph