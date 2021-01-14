'''
By Mar Balibrea Rull (@marbali8)

Resources used:

    (a) Megha Khosla, Vinay Setty, and Avishek Anand. "A Comparative Study for
    Unsupervised Network Representation Learning." IEEE Transactions on
    Knowledge and Data Engineering (2019).

    (b) https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/
'''

import networkx as nx
import numpy as np
import glob
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

# def maxvote(v, N, k) -> :

DATAPATH = 'data/'

def main(dataset: str, algorithm: str):
    # 1. Get the embedding, the graph and the lables

    print(dataset, algorithm)
    file_edges = open(DATAPATH + dataset + '/edges.csv', 'r')
    fe = sorted(glob.glob(DATAPATH + dataset + '/*embedding*' + algorithm +'*.npy'))[::-1][0]
    file_embedding = open(fe, 'rb')

    graph = nx.parse_edgelist(file_edges, delimiter = ',', create_using = nx.DiGraph)
    embedding = np.load(file_embedding, allow_pickle = True)
    embedding = embedding.item()

    labels = np.genfromtxt(DATAPATH + dataset + '/group-edges.csv', delimiter = ',', dtype = '<U20').T

    X, y = [embedding[node] for node in labels[0]], list(map(int, labels[1]))
    X, y = np.array(X), np.array(y)

    # 2. Separate train/test (80/20, 5-fold)

    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 8)

    # 3. Define model (multi-label classification using one-vs-rest logistic regression)
    # always: multi_class = 'ovr', max_iter = 1000, class_weight = 'balanced'
    # not bad:
    # - 'liblinear', 'l2'
    # - 'sag', 'l2'
    # - 'saga', != 'none'

    # others:
    # - 'sag', 'none'

    solvers = ['lbfgs', 'liblinear', 'sag', 'saga']
    f = open(DATAPATH + dataset + '/nodeclass_models_' + algorithm + '.csv', 'w')
    f.write('solver,avg_mf1,avg_Mf1,time,embedding_doc\n')
    for s in solvers:

        t_0 = time.time()
        model = LogisticRegression(multi_class = 'ovr', solver = s, max_iter = 1000, penalty = 'l2', class_weight = 'balanced')

        # 4. Train model
        mf1 = []
        Mf1 = []
        for train_index, test_index in kf.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            # print(model.n_iter_)
            # cross_validate(model, X, y, cv = 5, scoring = ('f1_micro', 'f1_macro'))

            # 5. Test model + compute metrics

            y_pred = model.predict(X_test)

            Mf1_model = f1_score(y_test, y_pred, average = 'macro')
            mf1_model = f1_score(y_test, y_pred, average = 'micro')

            Mf1.append(Mf1_model)
            mf1.append(mf1_model)

        a = np.mean(np.array(mf1), axis = 0)*100
        b = np.mean(np.array(Mf1), axis = 0)*100
        # print(a, b)
        f.write("{},{},{},{},{}\n".format(s, a, b, time.time()-t_0, fe.split('/')[-1]))

    f.close()

    # print(model.classes_)


    # 6. Compute Max-Vote baseline

    # maxvote = maxvote()

    # Mf1_maxvote = 0
    # mf1_maxvote = 0


if __name__ == "__main__":

    datasets = ['wikipedia/chameleon', 'wikipedia/squirrel', 'wikipedia/crocodile', 'BlogCatalog', 'Cora', 'PubMed']
    algorithms = ['deepwalk', 'node2vec']

    for d in datasets:
        for a in algorithms:

            main(d, a)
