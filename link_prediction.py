"""Made by Alessio Galatolo
"""
from math import exp
from networkx.classes.digraph import DiGraph
from networkx.classes.function import degree
from numpy import random
from sklearn.metrics import roc_auc_score
from numpy import inner, array
from math import floor

def link_probability(u, v):
    """gets the similarity of the two nodes

    Args:
        u (array-like): the embedding of u
        v (array-like): the embedding of v

    Returns:
        Number: the inner product of the embeddings
    """
    return inner(array(u), array(v))    

def link_prediction(graph, method, test_percentage=20):
    """Divides the given graph edges into training and test set
    then executes link prediction and returns a AUC score

    Args:
        graph (Graph or DiGraph): The graph to consider
        method (Callable): A method of embedding. The method MUST return a dictionary with pairs
                            {node (int): embedding (array-like)}
        test_percentage (int, optional): The percentage of the edges in the test set. Defaults to 20.
    """
    n_tests_pos = floor(graph.number_of_nodes() / 100 * test_percentage)
    test_edges = [] 
    test_negative_edges = []#will contain the negative edges
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

            #if graph is directed add negative edge
            if type(graph) is DiGraph:
                if (v, u) not in graph.edges:
                    test_negative_edges.pop()
                    test_negative_edges.append((v, u))
    n_tests_pos = len(test_edges)
    embedding = method(graph)

    ## ASSESS PREDICTION BASED ON TEST_SPLIT positive
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
    ## ASSESS PREDICTION BASED ON TEST_SPLIT negative
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
   
    return roc_auc_score(ground_truth, infered_truth)

if __name__ == "__main__":
    from Datasets.datasets import Datasets, get_graph
    from netmf.netmf import netmf
    print(link_prediction(get_graph(Datasets.Pubmed), netmf))
    