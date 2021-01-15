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
from sklearn.preprocessing import LabelEncoder
from Datasets.datasets import Datasets, get_graph, dataset2filename





def node_classification(dataset, algorithm):
    # 1. Get the embedding, the graph and the lables

    embedding = algorithm(get_graph(dataset))
    path = dataset2filename[dataset]

    labels = np.genfromtxt(path + 'group-edges.csv', delimiter = ',', dtype = '<U20').T
    
    try:
        int(labels[1][0])
    except ValueError:
        #labels need encoding
        le = LabelEncoder()
        labels[1] = le.fit_transform(labels[1])

    X, y = [embedding[int(node)] for node in labels[0]], list(map(int, labels[1]))
    X, y = np.array(X), np.array(y)


    props = [['l1', 'liblinear'], ['l1', 'saga'], ['l2', 'lbfgs'], ['l2', 'liblinear'], \
            ['l2', 'saga'], ['elasticnet', 'saga'], ['none', 'lbfgs'], ['none', 'saga']]
    props = [[p[0], p[1], b, k] for p in props for b in ['balanced', None] for k in ['strat', 'no']]

    for i, p in enumerate(props):

        if p[3] is not 'strat' and p[2] is not 'balanced':
            continue

        print(str(i) + '/' + str(len(props)-1), p)

        # 2. Separate train/test (80/20, 5-fold)

        if p[3] is 'strat':
            kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 8)
            splitting = kf.split(X, y)
        else:
            kf = KFold(n_splits = 5, shuffle = True, random_state = 8)
            splitting = kf.split(X)

        # 3. Define model (multi-label classification using one-vs-rest logistic regression)
        l1 = 0.5 if p[0] is 'elasticnet' else None
        t_0 = time.time()
        model = LogisticRegression(multi_class = 'ovr', solver = p[1], max_iter = 2000, \
                                    penalty = p[0], class_weight = p[2], l1_ratio = l1)

        # 4. Train model
        mf1 = []
        Mf1 = []
        for train_index, test_index in splitting:

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)

            # 5. Test model + compute metrics

            y_pred = model.predict(X_test)

            Mf1_model = f1_score(y_test, y_pred, average = 'macro')
            mf1_model = f1_score(y_test, y_pred, average = 'micro')

            Mf1.append(Mf1_model)
            mf1.append(mf1_model)

        a = np.mean(np.array(mf1), axis = 0)*100
        b = np.mean(np.array(Mf1), axis = 0)*100
        # print(a, b)
        c = "{},{},{},{},{},{},{},{}\n".format(p[0], p[1], p[2], p[3], a, b, time.time()-t_0, model.n_iter_, i)
        print(c)


if __name__ == "__main__":
    from netmf.netmf import netmf

    node_classification(Datasets.BlogCatalog, netmf)
