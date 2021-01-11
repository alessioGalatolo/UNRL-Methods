import networkx as nx
from node2vec import compute_transition_probabilities, generate_random_walks


PATH = '/Users/marbalibrea/Downloads/'
DATASET = PATH + 'Flickr/'

with open(DATASET + 'edges.csv', 'r') as data:

    graph = nx.parse_edgelist(data, delimiter = ',', create_using = nx.Graph)

    d_graph = compute_transition_probabilities(graph)
    print('TODO train / test')
    random_walks = generate_random_walks(d_graph)

    with open(DATASET + 'randomwalks.npy', 'wb') as f:

        np.save(f, random_walks)
