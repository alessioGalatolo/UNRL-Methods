"""Made by Alessio Galatolo

Resources used:
"""
from networkx import Graph
from math import exp
from sklearn.metrics import roc_auc_score
from numpy import inner, array

def link_probability(u, v):
    inner(array(u), array(v))    

def link_prediction(graph: Graph, method, test_percentage=20):
    test_edges = []
    for i, (u, v) in enumerate(list(graph.edges)):
        if i % 100 < test_percentage:
            #time to remove edges 

            #but only no isolated node is created
            if graph.degree[u] > 1 and graph.degree[v] > 1:
                graph.remove_edge(u, v)
                test_edges.append((u, v))

    embedding = method(graph)

    #do what you have to do - may not be the proper way but definitely the fastest
    link_probs = []
    for (u, v) in test_edges:
        if u in embedding and v in embedding:
            u_emb = embedding[u]
            v_emb = embedding[v]
        else:
            print("The node {u} or {v} was not found inside the embedding")
            exit()
        link_probs.append(link_probability(u_emb, v_emb))

    #normalized by the sigmoid function
    listMax = float(max(link_probs))
    
    link_probs = [1/(1+exp(-x / listMax)) for x in link_probs]
    print(roc_auc_score([1 for _ in range(len(link_probs))], link_probs))

if __name__ == "__main__":
    from Datasets.datasets import Datasets, get_graph
    from LINE1.line import line1

    link_prediction(get_graph(Datasets.WikiVote), line1)
    