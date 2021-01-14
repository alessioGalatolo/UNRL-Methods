from enum import Enum, auto
from networkx import Graph, is_weighted

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
    txt = auto()
    other = auto()
    
#given the dataset name will return the filename
dataset2filename = {Datasets.Twitter: "Datasets/twitter_combined.txt",
                    Datasets.BlogCatalog: "Datasets/blogcatalog.mat",
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
                    Datasets.Cora: "Datasets/cora/cora.cites", #todo
                    Datasets.Epinions: Formats.txt,
                    Datasets.Google: Formats.txt,
                    Datasets.WikiVote: Formats.txt,
                    Datasets.Pubmed: "Datasets/Pubmed-Diabetes/" #todo
                }

def get_graph(dataset: Datasets):
    #datasets in mat format
    filename = dataset2filename[dataset]
    format = dataset2format[dataset]
    graph = Graph() #the dataset to return 
    
    if format is Formats.mat:
        from scipy.io import loadmat
        data = loadmat(filename) 
        #data['network'] contains the graph, while data['group'] the labels
        #both are sparse matrix
        sparse_graph = data['network']
        for u, v, w in zip(*sparse_graph.nonzero(), sparse_graph.data):
            graph.add_edge(u, v, weight=w)
        #TODO use group as labels
    elif format is Formats.txt:
        from networkx import read_edgelist
        graph = read_edgelist(filename, create_using=Graph(), nodetype=int, data=(('weight',float),))
    elif format is Formats.other:
        pass

    #check if graph has weights
    if not is_weighted(graph):
        from networkx import set_edge_attributes
        set_edge_attributes(graph, values = 1, name = 'weight')

    return graph

if __name__ == "__main__":
    for dataset in Datasets:
        get_graph(dataset)