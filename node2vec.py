"""
By Mar Balibrea Rull (@marbali8)

Resources used:

    (a) A. Grover and J. Leskovec. node2vec: Scalable feature learning for
    networks. In SIGKDD, pages 855–864. ACM, 2016.

    (b) Megha Khosla, Vinay Setty, and Avishek Anand. "A Comparative Study for
    Unsupervised Network Representation Learning." IEEE Transactions on
    Knowledge and Data Engineering (2019).
"""

import numpy as np
from tqdm.auto import tqdm
import threading as th
from collections import defaultdict
import time
import random

import networkx as nx
import gensim

# 0. Define constants (from https://arxiv.org/pdf/1903.07902.pdf)

# random walk constants
WALK_LENGTH: int = 40 # Number of nodes in each walk
NUM_WALKS: int = 80 # Number of walks per source node
P: int = 0.25
Q: int = 2

# word2vec constants
N_NEGATIVE_SAMPLING: int = 5
NS_WINDOW = 10
EMBEDDING_DIMENSION: int = 128 # Embedding dimensions

# computation constants
N_THREADS: int = 4 # Number of threads for parallel execution
N_SAMPLES: int = int(1e6) # Number of training samples

DATAPATH = '/Users/marbalibrea/Downloads/Flickr/'

def compute_transition_probabilities(graph: nx.Graph) -> dict:

    """
    Step 1. Compute second order transition probabilities to later
    sample the next vertex in the walk. Eq. 4 in (b).

    :param graph: Unweighted (un)directed graph.
    :return: A simpler model of the graph.
    """

    t_0 = time.time()
    d_graph = defaultdict(dict)

    # print('Weighted graphs are not implemented')
    print('Computing transition probabilities')
    # u source node, v intermediate node, w destination node
    for source in tqdm(graph.nodes()):

        # Save neighbours
        neighbours = list(graph.neighbors(source))
        d_graph[source]['one-hop'] = neighbours

        for intermediate in neighbours:

            if 'prob' not in d_graph[intermediate]:
                d_graph[intermediate]['prob'] = dict()

            scores = list()

            # Calculate scores
            for destination in graph.neighbors(intermediate):

                p, q = P, Q
                if destination == source: # Reciprocity
                    t = 1 / p
                elif destination in graph[source]: # destination and source are one-hop neighbours
                    t = 1
                else:
                    t = 1 / q

                scores.append(t)

            # From scores to probabilities
            d_graph[intermediate]['prob'][source] = np.array(scores) / sum(scores)
            d_graph[source]['first'] = np.ones(len(neighbours)) / len(neighbours)

            with open(DATAPATH + 'transition_prob_i' + str(intermediate) + 's' + str(source) + '.npy', 'wb') as f:
                np.save(f, d_graph[intermediate]['prob'][source])
            with open(DATAPATH + 'transition_first_s' + str(source) + '.npy', 'wb') as f:
              np.save(f, d_graph[source]['first'])

    print(f"Transition probabilities computed. Total time was {time.time() - t_0}.")
    return d_graph

def rw_thread(d_graph: dict, thread_num: int):

    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    random.seed(thread_num)
    np.random.seed(thread_num)
    walks = list()

    print('Generating walks (Thread: {})'.format(thread_num))
    pbar = tqdm(total = NUM_WALKS)
    for n_walk in range(NUM_WALKS):

        pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Start walk
            walk = [source]

            walk_length = WALK_LENGTH

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get('one-hop', None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1: # For the first step

                    probabilities = d_graph[walk[-1]]['first']
                    walk_to = np.random.choice(walk_options, size = 1, p = probabilities)[0]
                else:

                    probabilities = d_graph[walk[-1]]['prob'][walk[-2]]
                    walk_to = np.random.choice(walk_options, size = 1, p = probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings to use word2vec

            walks.append(walk)

    pbar.close()
    global random_walks
    random_walks.append(walks)

def generate_random_walks(d_graph: dict):

    """
    Step 2. Generate the random walks for the skip-gram input.

    :param d_graph: A simpler model of the original graph.
    :return: List of walks. Each walk is a list of nodes.
    """

    print('Generating the random walks')
    t_0 = time.time()
    threads = [th.Thread(target = rw_thread, args = (d_graph, i)) for i in range(N_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with open(DATAPATH + 'random_walks.npy', 'wb') as f:

        global random_walks
        np.save(f, np.array(random_walks))
    print(f"Random walks generated. Total time was {time.time() - t_0}.")


def node2vec(graph: nx.Graph):

    """
    Main function to run node2vec algorithm.

    :param graph: The original graph (unweighted).

    """

    print("--------------------------------")
    print("Executing Node2Vec embedding method")
    print(f"Number of samples: {N_SAMPLES}")
    print(f"Negative samples: {N_NEGATIVE_SAMPLING}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print("--------------------------------")

    d_graph = compute_transition_probabilities(graph)
    global random_walks
    random_walks = list()
    generate_random_walks(d_graph)

    # global embedding
    # embedding = [(np.random.random() - 0.5) / EMBEDDING_DIMENSION \
    #     for _ in range(graph.number_of_edges() * EMBEDDING_DIMENSION)]
    #
    # print(f"Embedding process ended. Total time was {time.time() - t_0}")
    # return embedding

if __name__ == "__main__":

    with open(DATAPATH + 'edges.csv', 'r') as data:

        # CHANGE P AND Q AND nx.GRAPH/DIGRAPH AND DATAPATH!!!!
        graph = nx.parse_edgelist(data, delimiter = ',', create_using = nx.Graph)

        # embedding = node2vec(graph)
        node2vec(graph)

        # print(embedding)
