"""
Utilities used in graph convolutional layers.
"""
import numpy as np
from scipy import sparse as sp


def adjacency_matrix(edges: list, num_features: int, symmetric=True):
    """
    Build adjacency matrix `A` from the indicated `edges`.
    """
    # TODO: define: indexes in edges must be zero-based or starting at 1?
    # TODO: adjacency matrix will be symmetric when graph is <undirected>
    # TODO: edges from a vertex to itself (loops) are not allowed in <simple graphs>
    data = np.ones(len(edges))
    row_i, col_j = zip(*edges)

    if not max(row_i + col_j) < num_features:
        raise Exception(
            'Nodes in edges must be compatible with the number of features.')

    A = sp.coo_matrix((data, (row_i, col_j)),
                      shape=(num_features, num_features),
                      dtype=np.float32)

    if symmetric:
        A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    return A.todense()


def normalized_adjacency_matrix(a, method='kipf_welling'):
    """
    Normalizes the adjacency matrix according to indicated method.
    Defaults to 'Kipf & Welling' method.

    # Arguments
        a: Numpy array, the adjacency matrix.
        method: String, the normalization method. Either one of the following: `'kipf_welling'`.

    # Returns
        Numpy array, the normalized Adjacency matrix.

    # Raises
        ValueError: if `method` is invalid.

    """
    def norm_kipf_welling(a):
        """
        Normalize adjacency matrix as per Kipf and Welling (2017) method:
        https://arxiv.org/pdf/1609.02907.pdf

        This method was based on:
        - https://github.com/bknyaz/examples/blob/master/fc_vs_graph_train.py
        - https://github.com/tkipf/gcn
        """
        i = identity(a)
        a_hat = a + i
        d_hat = degree(a_hat, True)  # nodes degree (N,)
        d_hat_inv_sqrt = np.power(d_hat, -0.5)
        a_norm = d_hat_inv_sqrt * a_hat * d_hat_inv_sqrt  # (N, N)
        return a_norm

    methods = {'kipf_welling': norm_kipf_welling}
    return __execute_func_from_options(method, methods, a=a)


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalized_laplacian(a, method):
    """
    Normalizes the adjacency matrix according to indicated method.

    # Arguments
        a: Numpy array, the adjacency matrix.
        method: String, the normalization method. Either one of the following: `'symmetric'` or `'random_walk'`.

    # Raises
        ValueError: if `method` is invalid.

    """
    def symmetric_norm_laplacian(a):
        """ Symmetric normalized Laplacian
        """
        i = identity(a)
        d = degree(a, True)
        l_sym = i - (d**-1.) * a
        return l_sym

    def random_walk_norm_laplacian(a):
        """ Random walk normalized Laplacian 
        """
        i = identity(a)
        d = degree(a, True)
        l_rw = i - (d**-0.5) * a * (d**-0.5)
        return l_rw
    
    methods = {
        'symmetric': symmetric_norm_laplacian,
        'random_walk': random_walk_norm_laplacian
    }
    return __execute_func_from_options(method, methods, a=a)
    

def __execute_func_from_options(func_name:str, func_options:dict, **kwargs):
    """
    Execute a function by name from a list of options.
    """
    if func_name not in func_options:
        raise ValueError('{} is not a valid option.'.format(func_name))
    
    function = func_options.get(func_name)
    return function(**kwargs)
   

def laplacian(a):
    """
    Calculate the Laplacian matrix of the adjacency matrix A 
    """
    d = degree(a)
    return d - a


def degree(a, flatten=False):
    """
    Calculate the degree matrix of the adjacency matrix A 
    """
    assert (a.ndim > 0)

    d = a.sum(1)

    if not flatten:
        d = np.diag(d)
    return d


def identity(matrix):
    """
    Calculate the identity matrix of a matrix
    """
    assert (matrix.ndim > 0)
    return np.identity(matrix.shape[0])
