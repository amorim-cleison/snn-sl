"""
Utilities used in graph convolutional layers.
"""
import networkx as nx
from keras import backend as K


def degrees(graph):
    """
    Calculate the degrees
    """
    graph = get_graph(graph)
    d = [val for (node, val) in graph.degree()]
    return K.variable(d)


def identity(x):
    if x is nx.Graph:
        size = num_nodes(x)
    elif K.is_tensor(x):
        size = K.int_shape(x)[0]
    else:
        raise ValueError("Invalid parameter type.")
    return K.eye(size)


def adjacency_matrix(graph):
    """
    Calculate the adjacency matrix
    """
    graph = get_graph(graph)
    adj = nx.adjacency_matrix(graph)
    # FIXME: stop parsing sparse matrix to dense (Keras bug):
    return K.variable(adj.todense())


def get_graph(graph_data):
    if graph_data is nx.Graph:
        return graph_data
    else:
        return nx.Graph(graph_data)


def num_nodes(graph):
    graph = get_graph(graph)
    return graph.number_of_nodes()
