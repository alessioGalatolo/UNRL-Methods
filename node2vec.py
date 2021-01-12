'''
By Mar Balibrea Rull (@marbali8)

Resources used:

    (a) A. Grover and J. Leskovec. node2vec: Scalable feature learning for
    networks. In SIGKDD, pages 855–864. ACM, 2016.

    (b) Megha Khosla, Vinay Setty, and Avishek Anand. "A Comparative Study for
    Unsupervised Network Representation Learning." IEEE Transactions on
    Knowledge and Data Engineering (2019).

    (c) https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
'''

import numpy as np
from tqdm.auto import tqdm
import threading as th
from collections import defaultdict
import time
import random
import glob

import networkx as nx
import gensim.models as gem

# 0. Define constants (from https://arxiv.org/pdf/1903.07902.pdf)

# random walk constants
WALK_LENGTH: int = 40 # Number of nodes in each walk
NUM_WALKS: int = 80 # Number of walks per source node
P: int = 4
Q: int = 4

# word2vec constants
N_NEGATIVE_SAMPLING: int = 5
NS_WINDOW = 10
EMBEDDING_DIMENSION: int = 128 # Embedding dimensions

# computation constants
N_THREADS: int = 4 # Number of threads for parallel execution

DATAPATH = '/Users/marbalibrea/Downloads/aml/rw/'

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

            # with open(DATAPATH + 'transition_prob_i' + str(intermediate) + 's' + str(source) + '.npy', 'wb') as f:
            #     np.save(f, d_graph[intermediate]['prob'][source])
            # with open(DATAPATH + 'transition_first_s' + str(source) + '.npy', 'wb') as f:
            #     np.save(f, d_graph[source]['first'])

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
    pbar = tqdm(total = int(NUM_WALKS / N_THREADS))
    for n_walk in range(int(NUM_WALKS / N_THREADS)):

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

                if len(walk) == 1:  # For the first step

                    probabilities = d_graph[walk[-1]]['first']
                    walk_to = np.random.choice(walk_options, size = 1, p = probabilities)[0]
                else:

                    probabilities = d_graph[walk[-1]]['prob'][walk[-2]]
                    walk_to = np.random.choice(walk_options, size = 1, p = probabilities)[0]

                walk.append(walk_to)

            walk = np.array(list(map(str, walk)))  # Convert all to strings to use node2vec

            walks.append(walk)

    pbar.close()
    global random_walks
    random_walks.append(np.array(walks))

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
    # to retrieve: f = open(PATH, 'rb'), np.load(f, allow_pickle = True), f.close() (+ reshape)
    print(f"Random walks generated. Total time was {time.time() - t_0}.")

def node2vec(graph: nx.Graph = None, filename: str = None):

    """
    Main function to run node2vec algorithm.

    :param graph: The original graph (unweighted).
    :param filename: A filename with random walks already computed. The graph will be considered first.

    """

    global random_walks
    random_walks = list()
    if graph:

        print("--------------------------------")
        print("Executing Node2Vec embedding method")
        print(f"Number of samples: {NUM_WALKS * len(graph.nodes())}")
        print(f"Negative samples: {N_NEGATIVE_SAMPLING}")
        print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
        print("--------------------------------")

        d_graph = compute_transition_probabilities(graph)
        generate_random_walks(d_graph)
        np_rw = random_walks.copy()

    elif filename:

        t_0 = time.time()
        f = open(filename, 'rb')
        np_rw = np.load(f, allow_pickle = True)
        f.close()
        np_rw = np.reshape(np_rw, (np_rw.shape[0] * np_rw.shape[1], -1)) # thread flattening

        print("--------------------------------")
        print("Executing Node2Vec embedding method")
        print(f"Number of samples: {np_rw.shape[0]}")
        print(f"Negative samples: {N_NEGATIVE_SAMPLING}")
        print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
        print("--------------------------------")

        print(f"Filename was correctly loaded. Total time was {time.time() - t_0}.")

    else:
        print('No data was introduced!')
        return

    # t_0 = time.time()
    random_walks = []
    if type(np_rw[0][0]) == type(np.array([])):
        for rw in np_rw:
            random_walks.append([b for a in rw for b in a])
    else: # it's <class 'numpy.str_'>
        for rw in np_rw:
            random_walks.append([a for a in rw])
    # print(f"Pre-processing done. Total time was {time.time() - t_0}.")

    '''
    Step 3: Train the model.
    '''
    print('Starting Word2Vec')
    t_0 = time.time()

    # - to set: size to EMBEDDING_DIMENSION, window to NS_WINDOW,
    # sg to 1 (skip-gram), workers to N_THREADS, hs to 0 (negative sampling),
    # negative to N_NEGATIVE_SAMPLING (number of negative words)
    # - other: alpha / min_alpha (learning rate), epochs
    model = gem.Word2Vec(size = EMBEDDING_DIMENSION,  window = NS_WINDOW, \
                         min_count = 0, sg = 1, workers = N_THREADS, hs = 0, \
                         negative = N_NEGATIVE_SAMPLING, seed = 8)
    model.build_vocab(random_walks)
    model.train(random_walks, total_examples = model.corpus_count, epochs = model.epochs)
    embedding = {k: model[k] for k in model.wv.vocab.keys()}

    name = filename.split('/')[-1].split('.')[0]
    path = '/'.join(filename.split('/')[:-1]) + '/'
    model.save(path + name + '.model')
    # to retrieve: m = Word2Vec.load(PATH)

    f = open(path + name + '_embedding.npy', 'wb')
    np.save(f, embedding)
    f.close()
    # to retrieve: f = open(PATH, 'rb'), x = np.load(f, allow_pickle = True), x = x.item(), f.close()
    # model.init_sims(replace = True)

    print(f"Embedding process ended. Total time was {time.time() - t_0}.")
    return embedding

if __name__ == "__main__":

    # Option A: To calculate random walks.
    # IMPORTANT: change P, Q, nx.GRAPH/DIGRAPH (3 instances) and DATAPATH
    # with open(DATAPATH + 'edges.csv', 'r') as data:
    #
    #     graph = nx.parse_edgelist(data, delimiter = ',', create_using = nx.Graph)
    #
    #     node2vec(graph, None)

    # Option B: Random walks already computed. Only does word2vec.
    for file in glob.glob(DATAPATH + 'random_walks_cora_node.npy'):

        print('')
        print(file)
        embedding = node2vec(None, file)
        print('# words: ' + str(len(list(embedding.keys()))))
        print('')
