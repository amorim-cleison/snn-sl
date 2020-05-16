import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from keras import layers as l
from keras import backend as K

from third_party.keras_dgl import layers as dgl

from .architecture import Architecture

from third_party.keras.utils import graph_conv_utils
from third_party.keras.layers import GraphConvLSTM

import networkx as nx

class AttentionGraphConvLSTM(Architecture):
    """
    AGC-LSTM implementation as presented in:
    'An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition' 
    https://arxiv.org/abs/1902.09130

    Partial results:
    loss: 1.7926 - accuracy: 0.4483 - val_loss: 2.8913 - val_accuracy: 0.2273

    """

    data_format = 'channels_last'
    edges = [   (0, 1), (1, 2), (2, 16),
                (0, 3), (3, 4), (4, 5),
                (5, 6), (6, 7),
                (5, 8), (8, 9),
                (5, 10), (10, 11),
                (5, 12), (12, 13),
                (5, 14), (14, 15),
                (16, 17), (17, 18),
                (16, 19), (19, 20),
                (16, 21), (21, 22),
                (16, 23), (23, 24),
                (16, 25), (25, 26)  ]


    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)

    def build(self):
        return self.build_seq()
    """
        data_format = 'channels_last'

        input = l.Input(shape=self.input_shape)
        permute = l.Permute((2, 4, 3, 1))(
            input)  # Transformed to 'channels_last' (None, 60, 1, 27, 3)
        flatten = l.TimeDistributed(
            l.Flatten(data_format=data_format), name='flatten')(
                permute)  # Question: what is 'FC' layer here in paper?

        fa = l.Lambda(
            self.agc_lstm_feature_augmentation, name='augmentation')(flatten)

        mask = l.Masking(mask_value=0.)(fa)

        normalize = l.BatchNormalization(axis=-1)(mask)

        lstm_1 = l.LSTM(40, return_sequences=True)(normalize)

        tap_1 = l.AveragePooling1D(
            pool_size=3, strides=1, data_format=data_format,
            name='tap_1')(lstm_1)
        agc_lstm_1 = self.agc_lstm_cell(
            tap_1, filters=40, name='agc_lstm_1', return_sequences=True)

        tap_2 = l.AveragePooling1D(
            pool_size=3, strides=1, data_format=data_format,
            name='tap_2')(agc_lstm_1)
        agc_lstm_2 = self.agc_lstm_cell(
            tap_2, filters=40, name='agc_lstm_2', return_sequences=True)

        tap_3 = l.AveragePooling1D(
            pool_size=3, strides=1, data_format=data_format,
            name='tap_3')(agc_lstm_2)
        agc_lstm_3 = self.agc_lstm_cell(
            tap_3, filters=40, name='agc_lstm_3', return_sequences=False)

        fc = l.Dense(self.num_classes)(agc_lstm_3)
        activation = l.Activation('softmax')(fc)

        return super().build_model(inputs=input, outputs=activation)
    """

    def build_seq(self):
        # num_features = self.input_shape[-2]
        # tmp = np.ones(shape=(1, num_features, num_features))
        # graph_conv_tensor = K.constant(tmp, K.floatx())
        
        G = nx.from_edgelist(self.edges)
        adj = nx.adjacency_matrix(G)

        # adj = graph_conv_utils.adjacency_matrix(self.edges)
        norm_adj = graph_conv_utils.normalized_adjacency_matrix(adj)

        # FIXME: this is not a "normalized" adj_matrix, but part of a
        # specific formula from 'Kipf and Welling'
        # FIXME: stop parsing sparse matrix to dense (performance issue):
        norm_adj_tensor = K.variable(norm_adj.todense(), dtype=K.floatx()) 
                
        layers = [
            # l.Input(shape=self.input_shape),
            l.Permute((2, 4, 3, 1), input_shape=self.input_shape, name='permute'), # -> (60, 1, 27, 3) 
            l.Lambda(lambda x: K.squeeze(x, axis=2), name='squeeze'), 

            # ConvLSTM2D(10, (1, 2)),

            GraphConvLSTM(10, norm_adj_tensor, name='graph_conv_lstm', return_sequences=True),
            l.Flatten(data_format=self.data_format, name='flatten'),
            l.Dense(self.num_classes),
            l.Activation('softmax')
        ]
        return super().build_sequential(layers)
    
    def agc_lstm_cell(self,
                      input,
                      filters,
                      name='agc_lstm_cell',
                      return_sequences=True):
        # conv_1 = l.Conv1D(
        #     int(filters * 0.5), kernel_size=3,
        #     name='{}_conv_1'.format(name))(input)
        # lstm_1 = l.LSTM(
        #     filters,
        #     return_sequences=return_sequences,
        #     name='{}_lstm_1'.format(name))(conv_1)

        # TODO: replace with GraphConvLSTM

        num_features = input.shape[-2]
        # tmp = tf.zeros(shape=(1, num_features, num_features), dtype=tf.dtypes.float32)

        tmp = np.zeros(shape=(1, num_features, num_features))
        tmp[0][0][0] = 1

        # tmp = [[
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        #     [1, 0, 0, 0],
        # ]]
        graph_conv_tensor = K.variable(tmp, 'float32')
        # graph_conv_tensor = tf.zeros(
        #     shape=(1, num_features, num_features), dtype=tf.dtypes.float32)
        graph_conv_lstm = dgl.GraphConvLSTM(filters,
                                            graph_conv_tensor).call(input)

        return graph_conv_lstm


    def build_graph(self, edges, num_features):
        i = [i for (i, j) in edges]
        j = [j for (i, j) in edges]

        adj = sp.coo_matrix((np.ones(len(edges)), (i, j)), shape=(num_features, num_features), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        A = adj
        A = np.array(A.todense())

        # Build Graph Convolution filters
        SYM_NORM = True
        A_norm = self.preprocess_adj_numpy(A, SYM_NORM)
        num_filters = 2
        # graph_conv_filters = np.concatenate([A_norm, np.matmul(A_norm, A_norm)], axis=0)
        # graph_conv_filters = K.constant(graph_conv_filters)

        graph_conv_filters = np.expand_dims(A_norm, 0)
        graph_conv_filters = K.constant(graph_conv_filters)

        return graph_conv_filters


    def normalize_adj_numpy(self, adj, symmetric=True):
        if symmetric:
            d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
            a_norm = adj.dot(d).transpose().dot(d)
        else:
            d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
            a_norm = d.dot(adj)
        return a_norm

    def preprocess_adj_numpy(self, adj, symmetric=True):
        adj = adj + np.eye(adj.shape[0])
        adj = self.normalize_adj_numpy(adj, symmetric)
        return adj


    @tf.function
    def agc_lstm_feature_augmentation(self, input):
        """
        'Concatenate spatial feature and feature difference between two consecutive frames 
        to compose an augmented feature.'
        (2019 - Si et al - An Attention Enhanced Graph Convolutional 
        LSTM Network for Skeleton-Based Action Recognition)
        """
        # tf.config.experimental_run_functions_eagerly(True)
        # assert tf.executing_eagerly()

        # Difference between consecutive frames:
        rolled = tf.roll(input, shift=-1, axis=1)
        diff = tf.subtract(rolled, input)

        # Concatenate spatial feature and feature difference:
        concat = tf.concat([input, diff], axis=-1)

        return concat
