"""Made by Alessio Galatolo

Resources used:
"""
from networkx import Graph
from math import exp
from networkx.classes.digraph import DiGraph
from numpy import random
from sklearn.metrics import roc_auc_score
from numpy import inner, array
from math import floor

def link_probability(u, v):
    return inner(array(u), array(v))    

def link_prediction(graph, method, test_percentage=20):
    n_tests_pos = floor(graph.number_of_nodes() / 100 * test_percentage)
    test_edges = []
    test_negative_edges = []
    n_nodes = graph.number_of_nodes()
    nodes = list(graph.nodes)
    for i in range(n_tests_pos):
        rand = random.randint(n_nodes)
        u = nodes[rand]
        adj = graph.adj[u]
        for v in graph.nodes:
            if v != u and v not in adj:
                test_negative_edges.append((u, v))
                break
    n_tests_neg = len(test_negative_edges)

    edges = list(graph.edges)
    for i in range(n_tests_pos):
        (u, v) = edges[i * floor(100 / test_percentage)]
        #time to remove edges 
        #but only if no isolated node is created
        if graph.degree[u] > 1 and graph.degree[v] > 1:
            graph.remove_edge(u, v)
            test_edges.append((u, v))
            if type(graph) is DiGraph:
                if (v, u) not in graph.edges:
                    test_negative_edges.pop()
                    test_negative_edges.append((v, u))
    n_tests_pos = len(test_edges)
    embedding = method(graph)

    ## THIS IS PREDICTION FROM SCRATCH
    # links = []
    # edges = []
    # for u in graph.nodes:
    #     for v in graph.nodes:
    #         edges.append((u, v))
    #         if u in embedding and v in embedding:
    #             u_emb = embedding[u]
    #             v_emb = embedding[v]
    #             links.append(link_probability(u_emb, v_emb))
    #         else:
    #             print(f"The node {u} or {v} was not found inside the embedding")
    #             links.append(0)
    # predictions = sorted(zip(links, edges), key=lambda x: x[0], reverse=True)
    # predictions = predictions[:len(test_edges)]

    # #normalized by the sigmoid function
    # list_max = max(predictions, key=lambda x: x[0])[0]
    # predictions = [(1/(1+exp(-x / list_max)), (u, v)) for x, (u, v) in predictions]
    # ground_truth = []
    # infered_truth = []
    # for p, (u, v) in predictions[:len(test_edges)]:
    #     ground_truth.append(1 if (u, v) in test_edges or (v, u) in test_edges else 0)
    #     infered_truth.append(p)

    ## ASSESS PREDICTION BASED ON TEST_SPLIT
    ground_truth = [1 if x < n_tests_pos else 0 for x in range(n_tests_pos + n_tests_neg)]
    infered_truth = []
    for (u, v) in test_edges:
        if u in embedding and v in embedding:
            u_emb = embedding[u]
            v_emb = embedding[v]
            infered_truth.append(link_probability(u_emb, v_emb))
        else:
            print(f"The node {u} or {v} was not found inside the embedding")
            infered_truth.append(0)
    
    for (u, v) in test_negative_edges:
        if u in embedding and v in embedding:
            u_emb = embedding[u]
            v_emb = embedding[v]
            infered_truth.append(link_probability(u_emb, v_emb))
        else:
            print(f"The node {u} or {v} was not found inside the embedding")
            infered_truth.append(1)

    #normalized by the sigmoid function
    list_max = max(infered_truth)
    infered_truth = [1/(1+exp(-x / list_max)) for x in infered_truth]
   
    print(roc_auc_score(ground_truth, infered_truth))

if __name__ == "__main__":
    from Datasets.datasets import Datasets, get_graph
    from LINE1.line import line1
    from random_walk_based.deepwalk import deepwalk
    from random_walk_based.node2vec import node2vec

    link_prediction(get_graph(Datasets.Cora), line1)
    