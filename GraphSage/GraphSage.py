import torch
import numpy as np
import networkx as nx
import math
import random
from tqdm import tqdm

#notes:
#https://github.com/williamleif/GraphSAGE/issues/24
#Looking at this thread I decided to initialize the embedding with a 'two hot encoding' in order
#to have a less sparse representation. The paper also mention to use the degree of the node as initial embedding
#but it would resoult in a one-dimentional final embedding, which is not desiderable.
def method(G):
    K = 2 
    Q = 10
    epoch = 20
    s_size = [10,5]
    print("Reading the graph")
    #G = nx.read_edgelist(GraphFileName, delimiter=',')
    #nx.draw(G,with_labels = True)
    print("Generating Adj matrix, size:",len(G))
    A = nx.adjacency_matrix(G).todense()
    nodes_list = np.array(list(G.nodes()))
    N = A.shape[0]
    network_par = [N,128,128]#[N,128,128] #Initial size, hidden layer, embedding
    W = []
    print("Initializing parameters")
    for k in range(K):
        W.append(torch.nn.Parameter(torch.FloatTensor(network_par[k],network_par[k+1]),requires_grad=True))
        torch.nn.init.xavier_normal_(W[k])
    h0 =  torch.Tensor.float(torch.from_numpy(almostOldInitializeEmbedding(N,N)))

    print("Starting training")    
    Z=train(A,K,W,h0,epoch,network_par,Q,s_size)
    
    embedding = {}
    for i in range(N):
        embedding[nodes_list[i]] = Z[i].tolist()
    return embedding
    
def graphSage(G = None, GraphFileName = None):
    if G is None:
        if GraphFileName is None:
            GraphFileName = "Datasets/lastfm_asia/edges.csv"#"Data/cora.csv"#"Data/Pubmed-Diabetes.DIRECTED.cites.tab"##"Data/exampleTOY.dat"#"Data/cora.csv"
        #Graph loading
        print("Reading the graph")
        G = nx.read_edgelist(GraphFileName, delimiter=',')
    K = 2 
    Q = 10
    epoch = 20
    s_size = [10,5]
    print("clustering coeff:", nx.average_clustering(G), "T:", nx.transitivity(G))
    #nx.draw(G,with_labels = True)
    print("Generating Adj matrix, size:",len(G))
    A = nx.adjacency_matrix(G).todense()
    nodes_list = np.array(list(G.nodes()))
    N = A.shape[0]
    network_par = [N,128,128] #Initial size, hidden layer, embedding
    W = []
    print("Initializing parameters")
    for k in range(K):
        W.append(torch.nn.Parameter(torch.FloatTensor(network_par[k],network_par[k+1]),requires_grad=True))
        torch.nn.init.xavier_normal_(W[k])
    h0 =  torch.Tensor.float(torch.from_numpy(almostOldInitializeEmbedding(N,N)))

    print("Starting training")    
    Z=train(A,K,W,h0,epoch,network_par,Q,s_size)
    N=Z.shape[0]
    embedding = {}
    for i in range(N):
        embedding[nodes_list[i]] = Z[i].tolist()
    saveOnFile(embedding)
    return embedding
    
def saveOnFile(embedding):
    path = ""
    name = "feather"
    f = open(path + name + '_embedding.npy', 'wb')
    np.save(f, embedding)
    f.close()
    # to retrieve: f = open(PATH, 'rb'), x = np.load(f, allow_pickle = True), x = x.item(), f.close()

def train(A,K,W,h0,epoch,network_par,Q,s_size):
    N = A.shape[0]
    #indices = np.random.permutation(N)
    #train_set = list(indices)
    #n_batches = math.ceil(len(train_set)/256)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, W),lr=0.8)#8
    print("\tGenerating random Walks")
    positiveSample = generatePositiveSample(A)
    for e in tqdm(range(epoch)):
        #print("\t\tForward pass")
        Z = GraphSAGEmbeddingGen(A,W,h0,network_par,K,s_size)
        negativeSample = generateNegativeSample(A,Q)
        #print("\t\tComputing Loss")
        loss = torch.zeros(N)
        for i in range(N):
            Zu=Z[i]
            Zv=Z[np.random.choice(positiveSample[i])]
            Zvn =Z[negativeSample[i]]
            
            
            neg_loss = torch.nn.functional.cosine_similarity(Zu.view(1,-1), Zvn)
            neg_loss = Q*torch.mean(torch.log(torch.sigmoid(-neg_loss)), 0)
            pos_loss = torch.nn.functional.cosine_similarity(Zu.view(1,-1),Zv.view(1,-1))
            pos_loss = torch.log(torch.sigmoid(pos_loss))
            loss[i] = - pos_loss - neg_loss
        print('\n',torch.nn.functional.cosine_similarity(Z[0].view(1,-1), Z))
        loss.sum().backward(retain_graph=True)
        optimizer.step()
    return Z
    
def almostOldInitializeEmbedding(N,m):
    if N < m:
        return np.concatenate((np.identity(N), np.zeros((N,m-N))), axis=1)
    else:
        return np.identity(N) #WIll not work
def oldinitializeEmbedding(N):
    d = np.arange(N)
    m = math.ceil(np.log2(N))
    return ((d[:,None] & (1 << np.arange(m))) > 0).astype(int)

#Forward Pass
def GraphSAGEmbeddingGen(A,W,h0, network_par, K,s_size, Aggregator='mean'):
    N = A.shape[0]
    h=[]
    h.append(h0)
    for par in network_par[1:]:
        h.append(torch.zeros(N,par))   
    for k in range(K):
        for v in range(N):
            hN = Sample(A,v,h[k],K,s_size)
            out = MeanAggregator(h[k][v],hN,W[k])
            h[k+1][v] = out
    return h[K]

def MeanAggregator(h_self, h_neigh,W): # h_self should be 1xN and H_neigh should be MxN
    h=torch.cat((h_self.reshape(1,-1), h_neigh),0)
    h_mean=torch.mean(h,axis=0)
    h=torch.torch.nn.functional.relu(h_mean@W)
    return h

def Sample(A,v,h,K,s_size):# Only works for K=2
    
    original_v=v
    b=np.where(A[v]==1)[1]
    S1 = np.random.choice(b, size=s_size[0], replace=True)
    S = []
    #NOTE THIS MUST  BE FIXED IF K>2
    for v in S1:
        b=np.where(A[v]==1)[1]
        #b=b[b!=original_v]
        S.append(np.random.choice(b, size=s_size[1], replace=True))
    S=np.array(S).flatten()
    return h[S]

def convertToIdx(path, node_array):
    return 0

def randomWalk(A,start,walkLength):
    path = []
    random_node = start # Graph has node that go from 1 to N (included)
    for i in range(walkLength):
        list_for_nodes = list(np.where(A[random_node]==1)[1])
        if len(list_for_nodes)==0:# if random_node having no outgoing edges
            return path
        else:
            random_node = random.choice(list_for_nodes) #choose a node randomly from neighbors
            path.append(random_node)
            
    return path


def generatePositiveSample(A):
    positiveSample = []
    N= A.shape[0]
    for i in range(N):
        positiveSample_n = []
        for j in range(50):
            walk = randomWalk(A,i,5)
            walk = [x for x in walk if x != i] # remove itself from the walk
            positiveSample_n.extend(walk)
        positiveSample.append(positiveSample_n)
    return positiveSample

def generateNegativeSample(A,Q):
    negativeSample = []
    N= A.shape[0]
    for v in range(N):
        negativeSample_k = []
        while len(negativeSample_k)<Q:
            u = np.random.randint(0,N)
            if((A[v,u]==0) and u!=v ):
                negativeSample_k.append(u)
        negativeSample.append(negativeSample_k)
    return negativeSample


if __name__ == "__main__":
    graphSage()
