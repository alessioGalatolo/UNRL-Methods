import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import networkx as nx

def vol(A):
    """Calculate volume of a graph
    Args:
        A (sparse matrix): Adjacency matrix in sparse format
    """
    return np.sum(np.square(A))

def create_embedding(M, dimension):
    """Create embedding from deepwalk matrix
    Args:
        M (sparse matrix): Deepwalk matrix in sparse format
        dimension (int): dimension of the embedding
    """
    M.data[M.data < 1] = 1
    M.data = np.log(M.data)
    u, s, v = linalg.svds(M, dimension, return_singular_vectors="u")
    return u.dot(np.diag(np.sqrt(s)))

def deepwalk_matrix(P, Dinv, volG, window, b):
    """Calculate deepwalk matrix
    Args:
        P (sparse matrix): equals to matrix multiplication of Dinv and the adjacency matrix
        Dinv (sparse matrix): inverse of eigenvalues of adjacency matrix
        volG (float): volume of the graph
        window (int): Window size of the method
        b (float): parameter for negative sampling
    """
    P_power = sparse.identity(Dinv.shape[0])
    P_sum = np.zeros_like(P)
    for _ in range(window):
        P_power = sparse.coo_matrix(P_power.dot(P), shape = Dinv.shape)
        P_sum = P_sum + P_power
    M = (volG / (window * b)) * sparse.coo_matrix(P_sum.dot(Dinv), shape = Dinv.shape)
    return M

def small_embedding(A, window, b, dimension):
    """Calculate embedding for small windows
    Args:
        A (sparse matrix): Adjacency matrix in sparse format
        window (int): Window size of the method
        b (float): parameter for negative sampling
        dimension (int): dimension of the embedding
    """
    volG = vol(A)

    diagD = A.sum(axis=1)
    diagD = diagD.flatten().tolist()[0]
    diagDinv = np.array([1/Dj for Dj in diagD])
    Dinv = sparse.diags(diagDinv)
    P = sparse.coo_matrix(Dinv.dot(A))

    M = deepwalk_matrix(P, Dinv, volG, window, b)
    return create_embedding(M, dimension)

def approximate_deepwalk_matrix(L, U, Drootinv, volG, window, b):
    """Approximation of the deepwalk matrix for large windows
    Args:
        L (array): array of eigenvalues of the laplacian
        U (matrix): matrix of eigenvectors of the laplacian
        Drootinv (sparse matrix): inverse root of eigenvalues of adjacency matrix
        volG (float): volume of the graph
        window (int): Window size of the method
        b (float): parameter for negative sampling
    """
    L_power = L
    L_sum = np.zeros_like(L)
    for _ in range(window - 1):
        L_sum += L_power
        L_power = np.multiply(L_power, L)
    L_sum += L_power
    
    L_sum = np.diag(L_sum)
    
    DrootinvU = Drootinv.dot(U)
    
    M = (volG / (window * b)) * sparse.coo_matrix(DrootinvU.dot(L_sum).dot(DrootinvU.T), shape = Drootinv.shape)

    return M

def large_embedding(A, window, b, dimension, h):
    """Calculate embedding for large windows
    Args:
        A (sparse matrix): Adjacency matrix in sparse format
        window (int): Window size of the method
        b (float): parameter for negative sampling
        dimension (int): dimension of the embedding
        h (int): dimension of the intermediary image
    """
    volG = vol(A)
    diagD = A.sum(axis=1)
    diagD = diagD.flatten().tolist()[0]
    diagDrootinv = np.array([1/np.sqrt(Dj) for Dj in diagD])
    Drootinv = sparse.diags(diagDrootinv)
    laplacian = Drootinv.dot(A).dot(Drootinv)
    L, U = linalg.eigsh(laplacian, h)
    
    M = approximate_deepwalk_matrix(L, U, Drootinv, volG, window, b)
    return create_embedding(M, dimension)

def netmf(G, large = True, window = 10, b = 1, dimension = 128, h = 256):
    """Compute nf embedding of a graph
    Args:
        G (graph): Initial graph
        large (int): large or true version of the embedding
        window (int): Window size of the method
        b (float): parameter for negative sampling
        dimension (int): dimension of the embedding
        h (int): dimension of the intermediary image
    """
    G = G.to_undirected()
    A = nx.adjacency_matrix(G)
    if large :
        matrix = large_embedding(A, window, b, dimension, h)
    else :
        matrix = small_embedding(A, window, b, dimension)
    return {list(G.nodes)[i]: x.tolist() for i, x in enumerate(matrix)}


if __name__ == "__main__":
    from Datasets.datasets import Datasets, get_graph
    print(netmf(get_graph(Datasets.Pubmed)))