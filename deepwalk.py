'''
By Mar Balibrea Rull (@marbali8)

Resources used:

    (a) B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning
    of social representations. In SIGKDD, pages 701–710. ACM, 2014.

    (b) Megha Khosla, Vinay Setty, and Avishek Anand. "A Comparative Study for
    Unsupervised Network Representation Learning." IEEE Transactions on
    Knowledge and Data Engineering (2019).
'''

import numpy as np
from tqdm.auto import tqdm
import threading as th
import time
import glob

import networkx as nx
import gensim.models as gem

# 0. Define constants (from https://arxiv.org/pdf/1903.07902.pdf)

# random walk constants
WALK_LENGTH: int = 40 # Number of nodes in each walk
NUM_WALKS: int = 80 # Number of walks per source node

# word2vec constants
WINDOW = 10
EMBEDDING_DIMENSION: int = 128 # Embedding dimensions

# computation constants
N_THREADS: int = 4 # Number of threads for parallel execution

DATAPATH = '/Users/marbalibrea/Downloads/aml/rw/'


def rw_thread(graph: nx.Graph, thread_num: int):

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
        shuffled_nodes = list(graph.nodes())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Start walk
            walk = [source]

            walk_length = WALK_LENGTH

            # Perform walk
            while len(walk) < walk_length:

                walk_options = graph.neighbors(walk[-1])

                # Skip dead end nodes
                if not walk_options:
                    break

                walk_to = np.random.choice(walk_options, size = 1)[0]
                walk.append(walk_to)

            walk = np.array(list(map(str, walk)))  # Convert all to strings to use node2vec

            walks.append(walk)

    pbar.close()
    global random_walks
    random_walks.append(np.array(walks))

def generate_random_walks(graph: nx.Graph):

    """
    Step 1. Generate the random walks for the skip-gram input.

    :param graph: Unweighted (un)directed graph.
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

def deepwalk(graph: nx.Graph):

    """
    Main function to run deepwalk algorithm.

    :param graph: The original graph (unweighted).

    """

    global random_walks
    random_walks = list()

    print("--------------------------------")
    print("Executing DeepWalk embedding method")
    print(f"Number of samples: {NUM_WALKS * len(graph.nodes())}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print("--------------------------------")

    generate_random_walks(graph)

    '''
    Step 2: Train the model.
    '''
    print('Starting Word2Vec')
    t_0 = time.time()

    # - to set: size to EMBEDDING_DIMENSION, window to WINDOW,
    # sg to 1 (skip-gram), workers to N_THREADS, hs to 1 (hierarchical softmax)
    # - other: alpha / min_alpha (learning rate), epochs
    # model = gem.Word2Vec(size = EMBEDDING_DIMENSION, window = WINDOW, \
    #                      min_count = 0, sg = 1, workers = N_THREADS, hs = 1, \
    #                      seed = 8)
    # model.build_vocab(random_walks)
    # model.train(random_walks, total_examples = model.corpus_count, epochs = model.epochs)
    # embedding = {k: model[k] for k in model.wv.vocab.keys()}
    #
    # name = filename.split('/')[-1].split('.')[0]
    # path = '/'.join(filename.split('/')[:-1]) + '/'
    # model.save(path + name + '.model')
    # # to retrieve: m = Word2Vec.load(PATH)
    #
    # f = open(path + name + '_embedding.npy', 'wb')
    # np.save(f, embedding)
    # f.close()
    # to retrieve: f = open(PATH, 'rb'), x = np.load(f, allow_pickle = True), x = x.item(), f.close()
    # model.init_sims(replace = True)

    embedding = None
    print(f"Embedding process ended. Total time was {time.time() - t_0}.")
    return embedding

if __name__ == "__main__":

    # Option A: To calculate random walks.
    # IMPORTANT: change nx.GRAPH/DIGRAPH (4 instances) and DATAPATH
    with open(DATAPATH + 'edges.csv', 'r') as data:

        graph = nx.parse_edgelist(data, delimiter = ',', create_using = nx.Graph)

        deepwalk(graph)
