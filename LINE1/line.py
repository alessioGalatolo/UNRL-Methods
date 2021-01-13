"""Made by Alessio Galatolo

Resources used:
    LINE: Large-scale Information Network Embedding by Jian Tang et al.
    https://github.com/tangjianpku/LINE
"""
from networkx import Graph, read_edgelist
from networkx.classes.function import is_weighted, set_edge_attributes
import numpy as np
from math import exp
import threading
from time import time

#constants (values taken from original implementation)
N_THREADS = 4 #number of thread for asynchronous gradient descent
N_SAMPLES = 1000000 #number of samples for training
N_NEGATIVE_SAMPLING = 4
EMBEDDING_DIMENSION = 2 #the dimension of the embedding
NEG_SAMPLING_POWER = 0.75 #the power to elevate the negative samples
NEG_TABLE_SIZE = 100000000
SIGMOID_TABLE_SIZE = 1000
SIGMOID_BOUND = 6 #max and min value for sigmoid

#global variables
prob, alias = None, None
negative_table = [0 for _ in range(NEG_TABLE_SIZE)]
sigmoid_table = [0 for _ in range(SIGMOID_TABLE_SIZE)]
#the result of the embedding of the vertices
embedding = None
#for learning rate
initial_rho = 0.025 #value from original implementation
rho = initial_rho
current_sample_count = 0


#generate various tables for faster computation

def generate_sigmoid_table():
    """Compute and store common sigmoid values
    """
    global sigmoid_table
    for i in range(SIGMOID_TABLE_SIZE):
        x =  2 * SIGMOID_BOUND * i / SIGMOID_TABLE_SIZE - SIGMOID_BOUND
        sigmoid_table[i] = 1 / (1 + exp(-x))

    
def generate_negative_table(graph: Graph):
    """Get negative vertex samples according to vertex degrees

    Args:
        graph (Graph): The original graph
    """
    global negative_table
    sum = 0
    n_nodes = graph.number_of_nodes()
    por, cur_sum, vid = 0, 0, 0
    for (node, d) in graph.degree:
        sum += d ** NEG_SAMPLING_POWER
    for i in range(NEG_TABLE_SIZE):
        if (i + 1) / NEG_TABLE_SIZE > por:
            while True:
                try:
                    cur_sum += graph.degree[vid % n_nodes] ** NEG_SAMPLING_POWER
                    por = cur_sum / sum
                    vid += 1
                    break
                except KeyError:
                    vid += 1
        negative_table[i] = vid - 1

def generate_alias_table(graph: Graph):
    """ generate alias table for constant time edge sampling.
        Alias table method from: Reducing the sampling complexity of topic models. 
        by A. Q. Li et al.
    Args:
        graph (Graph): the graph from which to generate the alias table
    """
    global prob, alias
    n_edges = graph.number_of_edges()
    prob, alias = [0 for _ in range(n_edges)], [0 for _ in range(n_edges)]

    #total sum of weights in graph
    weight_sum = 0
    #the graph has weights
    for _, _, w in graph.edges.data("weight"):
        weight_sum += w
    norm_prob = [weight * n_edges / weight_sum for _, _, weight in graph.edges.data("weight")]

    small_block = []
    large_block = []
    for i in range(len(norm_prob) - 1, -1, -1):
        if norm_prob[i] < 1:
            small_block.append(i)
        else: 
            large_block.append(i)
    
    while len(small_block) > 0 and len(large_block) > 0:
        c_sb = small_block.pop()
        c_lb = large_block.pop()
        prob[c_sb] = norm_prob[c_sb]
        alias[c_sb] = c_lb
        norm_prob[c_lb] += norm_prob[c_sb] - 1
        if norm_prob[c_lb] < 1:
            small_block.append(c_lb)
        else:
            large_block.append(c_lb)
    
    while len(small_block) > 0:
        prob[small_block.pop()] = 1
    while len(large_block) > 0:
        prob[large_block.pop()] = 1


#utility functions

def fast_sigmoid(x):
    """Compute sigmoid of x reusing store values

    Args:
        x (Number): A number (expected to be between between the sigmoid bound)

    Returns:
        Number: Value in [0, 1]
    """
    if x > SIGMOID_BOUND:
        return 1
    elif x < -SIGMOID_BOUND:
        return 0
    k = (x + SIGMOID_BOUND) * SIGMOID_TABLE_SIZE / SIGMOID_BOUND / 2
    return sigmoid_table[k]

def Rand(seed):
    """Fastly generate a random integer. Code from original source.

    Args:
        seed (Number): the seed for the random generator

    Returns:
        (Number, Number): An updated seed and a (pseudo)random number
    """
    seed = seed * 25214903917 + 11
    return seed, (seed >> 16) % NEG_TABLE_SIZE


#graph related functions

def sample_edge(graph, rand1, rand2):
    """sample a random edge

    Args:
        graph (Graph): the original graph
        rand1 (Number): random from normal distribution
        rand2 (NUmber): random from normal distribution

    Returns:
        Number: the index of the sampled edge
    """
    k = int(rand1 * graph.number_of_edges())
    return k if rand2 < prob[k] else alias[k]

def update(lu, lv, error_vector, label):
    """Update embedding

    Args:
        lu (Number): The index of the source node
        lv (Number): The index of the context node
        error_vector (array-like of Number): The 
        label ([type]): [description]
    """
    x, g = 0, 0
    for i in range(EMBEDDING_DIMENSION):
        x += embedding[lu + i] * embedding[lv + i]
    g = (label - fast_sigmoid(x)) * rho
    for i in range(EMBEDDING_DIMENSION):
        error_vector[i] += g * embedding[lv + i]
        embedding[lv + i] += g * embedding[lu + i]


def line_thread(seed, graph):
    """This is the function used for asynchronous stochastic gradient decent

    Args:
        seed (Nuber): It is the id of the thread, will be used as a random seed
        graph (Graph): the original graph
    """
    print(f"Thread {seed} has been started")
    global embedding
    count = 0
    last_print = 0 #every now and then print whats happening
    while count <= N_SAMPLES  / N_THREADS + 2:
        print("Done cycle ", count)
        #give sign of life and update rho
        if count - last_print > 1e4:
            current_sample_count = count - last_print
            last_print = count
            print(f"Thread {seed} had done {count} iterations and is willing to do more")
            rho = initial_rho * (1 - current_sample_count / (N_SAMPLES - 1))
            if rho < initial_rho * 1e-4:
                rho = initial_rho * 1e-4
            
        #sample an edge
        edge = sample_edge(graph, np.random.random(), np.random.random())
        u, v = list(graph.edges)[edge]
        lu = u * EMBEDDING_DIMENSION
        lv = v * EMBEDDING_DIMENSION

        error_vector = [0 for _ in range(EMBEDDING_DIMENSION)]
        target, label = 0, 0
        for i in range(N_NEGATIVE_SAMPLING + 1):
            if i == 0:
                target = v
                label = 1
            else:
                seed, rand_number = Rand(seed)
                target = negative_table[rand_number]
                label = 0
            lv = target * EMBEDDING_DIMENSION
            update(lu, lv, error_vector, label)

        for i, val in enumerate(error_vector):
            embedding[lu + i] = val 
        count += 1

def line1(graph: Graph):
    global embedding, N_SAMPLES 
    N_SAMPLES = min(N_SAMPLES, graph.number_of_edges())
    print("--------------------------------")
    print("Executing LINE-1 embedding method")
    print(f"Number of samples: {N_SAMPLES}")
    print(f"Negative samples: {N_NEGATIVE_SAMPLING}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"Initial rho: {initial_rho}")
    print("--------------------------------")

    #check if graph has weights
    if not is_weighted(graph):
        set_edge_attributes(graph, values = 1, name = 'weight')
    # embedding = {}
    embedding = [(np.random.random() - 0.5) / EMBEDDING_DIMENSION \
        for _ in range(graph.number_of_edges() * EMBEDDING_DIMENSION)]
    generate_alias_table(graph)
    generate_negative_table(graph)
    generate_sigmoid_table()

    t_0 = time()
    threads = [threading.Thread(target=line_thread, args=(i, graph)) for i in range(N_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    print(f"Embedding process ended. Total time was {time() - t_0}")
    return {i: e for i, e in enumerate(embedding)} #return as dictionary
    return embedding

#test this method
if __name__ == "__main__":
    graph = read_edgelist("./Datasets/WikiVote.txt", create_using=Graph(), nodetype=int, data=(('weight',float),))

    embedding = line1(graph)

    print(embedding)