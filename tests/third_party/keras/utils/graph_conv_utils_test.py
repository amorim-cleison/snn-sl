import pytest
from third_party.keras.utils import graph_conv_utils
import numpy as np
import numpy.testing as nptest

a = np.asarray([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

i = np.asarray([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

d = np.asarray([
    [2, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 2]
])

l = d - a

x = np.asarray([
    [2, 1,  5,  9],
    [4, 6, 10,  8],
    [6, 9, 15, 12]
])

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_adjaceny_matrix():
    # graph = graph_conv_utils.get_graph(edges)
    a_calc = graph_conv_utils.adjacency_matrix(edges)
    __assert_matrices_equals(a_calc, a)


# def test_normalize_adjacency_matrix():
#     a_norm_calc = graph_conv_utils.normalized_adjacency_matrix(a)
#     a_norm = np.asarray([
#                 [0.333, 0.333,     0, 0.333],
#                 [0.333, 0.333, 0.333,     0],
#                 [    0, 0.333, 0.333, 0.333],
#                 [0.333,     0, 0.333, 0.333]
#             ])
#     __assert_matrices_equals(a_norm_calc, a_norm)


# def test_laplacian():
#     l_calc = graph_conv_utils.laplacian(a)
#     __assert_matrices_equals(l_calc, l)


# def test_degree():
#     d_calc = graph_conv_utils.degree(a)
#     __assert_matrices_equals(d_calc, d)


# def test_identity():
#     i_calc = graph_conv_utils.identity(a)
#     __assert_matrices_equals(i_calc, i)


# def test_laplacian_sym_norm():
#     l_sym_calc = graph_conv_utils.normalized_laplacian(a, 'symmetric')
#     l_sym = np.asarray([
#                 [1,    -0.5,    0, -0.5],
#                 [-0.5,    1, -0.5,    0],
#                 [   0, -0.5,    1, -0.5],
#                 [-0.5,    0, -0.5,    1]
#             ])
#     __assert_matrices_equals(l_sym_calc, l_sym)


# def test_laplacian_random_walk():
#     l_rw_calc = graph_conv_utils.normalized_laplacian(a, 'random_walk')
#     l_rw = np.asarray([
#                 [1,    -0.5,    0, -0.5],
#                 [-0.5,    1, -0.5,    0],
#                 [   0, -0.5,    1, -0.5],
#                 [-0.5,    0, -0.5,    1]
#             ])
#     __assert_matrices_equals(l_rw_calc, l_rw)


def __assert_matrices_equals(x, y):
    nptest.assert_array_almost_equal_nulp(x, y, 3)


if __name__ == '__main__':
    pytest.main([__file__])
