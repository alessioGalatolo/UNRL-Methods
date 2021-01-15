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
    WikiVote = auto()
    Pubmed = auto()

class Formats(Enum):
    mat = auto()
    txt = auto() #all graphs with this format are directed
    other = auto()
    
#given the dataset name will return the filename
dataset2filename = {Datasets.Twitter: "Datasets/twitter_combined.txt",
                    Datasets.BlogCatalog: "Datasets/blogcatalog/",
                    Datasets.YouTube: "Datasets/youtube.mat",
                    Datasets.Flickr: "Datasets/flickr.mat",
                    Datasets.CitHepPh: "Datasets/Cit-HepPh.txt",
                    Datasets.Reddit: "Datasets/reddit/", #todo
                    Datasets.Cora: "Datasets/cora/cora.cites", #todo
                    Datasets.Epinions: "Datasets/soc-Epinions1.txt",
                    Datasets.Google: "Datasets/web-Google.txt",
                    Datasets.WikiVote: "Datasets/WikiVote.txt",
                    Datasets.Pubmed: "Datasets/Pubmed-Diabetes/" #todo
                    }

#given the dataset name will return the format (.mat, list of edges or others)
dataset2format = {Datasets.Twitter: Formats.txt,
                    Datasets.BlogCatalog: Formats.mat,
                    Datasets.YouTube: Formats.mat,
                    Datasets.Flickr: Formats.mat,
                    Datasets.CitHepPh: Formats.txt,
                    Datasets.Reddit: "Datasets/reddit/", #todo
                    Datasets.Cora: Formats.txt, #todo
                    Datasets.Epinions: Formats.txt,
                    Datasets.Google: Formats.txt,
                    Datasets.WikiVote: Formats.txt,
                    Datasets.Pubmed: "Datasets/Pubmed-Diabetes/" #todo
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
                            Datasets.WikiVote: DiGraph(),
                            Datasets.Pubmed: DiGraph()
                        }

def get_graph(dataset: Datasets):
    #datasets in mat format
    filename = dataset2filename[dataset]
    format = dataset2format[dataset]
    graph = dataset2directionality[dataset] #the graph to return 
    
    if format is Formats.mat:
        from pandas import read_csv
        data = read_csv(filename + "edges.csv") 
        
        for u, v in data.values:
            graph.add_edge(u, v, weight=1)
        #TODO use group as labels
    elif format is Formats.txt:
        from networkx import read_edgelist
        graph = read_edgelist(filename, create_using=graph, nodetype=int, data=(('weight',float),))
    elif format is Formats.other:
        pass

    #check if graph has weights - if not add uniform
    if not is_weighted(graph):
        from networkx import set_edge_attributes
        set_edge_attributes(graph, values = 1, name = 'weight')

    return graph

if __name__ == "__main__":
    for dataset in Datasets:
        get_graph(dataset)