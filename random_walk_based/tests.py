# COMPROVATIONS WITH RANDOM_WALKS

# import numpy as np
#
# DATAPATH = '/Users/marbalibrea/Downloads/Flickr/'
#
# with open(DATAPATH + 'random_walks.npy', 'rb') as f:
#     rw = np.load(f, allow_pickle = True)
#
#
# print(rw.shape) # (4, x), 4 threads x (num_nodes (?) · num_walks (80) / 4 threads), num_nodes = len(graph.nodes())
# rwf = np.reshape(rw, (rw.shape[0] * rw.shape[1], -1)) # (x·4,)
# print(rwf.shape)
# print(np.all(rwf[rw.shape[1]] == rw[1][0]))
#
# lengths = []
# for i, r in enumerate(rwf):
#
#     lengths.append(r.shape[0])
#
# print(np.mean(lengths), np.max(lengths), np.min(lengths))
# (unique, counts) = np.unique(lengths, return_counts = True)
# maxs = np.argsort(counts)[::-1][:5]
# frequencies = np.asarray((unique[maxs], counts[maxs])).T
# print(frequencies)

# REFORMAT GROUP-EDGES WITH NAMES INSTEAD OF NUMBERS

# import numpy as np
#
# labels = np.genfromtxt('data/Cora/group-edges.csv', delimiter = ',', dtype = '<U20').T
# classes = np.unique(labels[1])
#
# f = open('data/Cora/group-edges-new.csv','w')
# g = open('data/Cora/group-edges-mapping.csv','w')
#
# for n in range(labels.shape[1]):
#
#     c = np.where(classes == labels[1][n])[0][0]
#     # f.write(str(labels[0][n]) + ',' + str(c) + '\n')
#     f.write("{},{}\n".format(labels[0][n], c))
#
#
# f.close()
#
# for i, c in enumerate(classes):
#
#     # g.write(str(i) + ',' + c + '\n')
#     g.write("{},{}\n".format(i, c))
#
# g.close()


# COSINE SIMILARITY OF EMBEDDINGS (with embedding 0, e_np[0])

f = open('/Users/marbalibrea/Downloads/Cora/embedding_deepwalk.npy', 'rb')
x = np.load(f, allow_pickle = True)
x = x.item()
f.close()

embedding_list = [x[a] for a in x.keys()]
e_np = np.array(embedding_list)
cossim = []
for e in e_np:
  cossim.append(e_np[0] @ e)
np.array(list(x.keys()))[np.argsort(cossim)[::-1]]
