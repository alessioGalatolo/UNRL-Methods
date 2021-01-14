"""Made by Alessio Galatolo

Resources used:
"""
from networkx import Graph
from math import exp
from sklearn.metrics import roc_auc_score
from numpy import inner, array

def link_probability(u, v):
    return inner(array(u), array(v))    

def link_prediction(graph: Graph, method, test_percentage=20):
    test_edges = []
    for i, (u, v) in enumerate(list(graph.edges)):
        if i % 100 < test_percentage:
            #time to remove edges 
            #but only if no isolated node is created
            if graph.degree[u] > 1 and graph.degree[v] > 1:
                graph.remove_edge(u, v)
                test_edges.append((u, v))

    embedding = method(graph)

    links = []
    edges = []
    for u in graph.nodes:
        for v in graph.nodes:
            edges.append((u, v))
            if u in embedding and v in embedding:
                u_emb = embedding[u]
                v_emb = embedding[v]
                links.append(link_probability(u_emb, v_emb))
            else:
                print("The node {u} or {v} was not found inside the embedding")
                links.append(0)
    predictions = sorted(zip(links, edges), key=lambda x: x[0], reverse=True)
    predictions = predictions[:len(test_edges)]

    #normalized by the sigmoid function
    list_max = max(predictions, key=lambda x: x[0])[0]
    predictions = [(1/(1+exp(-x / list_max)), (u, v)) for x, (u, v) in predictions]
    ground_truth = []
    infered_truth = []
    for p, (u, v) in predictions[:len(test_edges)]:
        ground_truth.append(1 if (u, v) in test_edges else 0)
        infered_truth.append(p)

    print(roc_auc_score(ground_truth, infered_truth))

if __name__ == "__main__":
    from Datasets.datasets import Datasets, get_graph
    from LINE1.line import line1
    from random_walk_based.deepwalk import deepwalk

    link_prediction(get_graph(Datasets.Cora), deepwalk)
    