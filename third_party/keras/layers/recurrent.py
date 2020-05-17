import warnings

import numpy as np
from keras import activations
from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.engine.base_layer import InputSpec
from keras.layers import Layer
from keras.layers.recurrent import (RNN, _generate_dropout_mask,
                                    _standardize_args)
from keras.utils.generic_utils import has_arg, to_list, transpose_shape
from networkx import Graph, adjacency_matrix
from third_party.keras.utils.graph_conv_utils import normalized_adjacency_matrix

# TODO: remove 
import networkx as nx
from third_party.keras.utils import graph_conv_utils as utils

class GraphConvRNN(RNN):

    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if unroll:
            raise TypeError('Unrolling isn\'t possible with '
                            'convolutional RNNs.')
        if isinstance(cell, (list, tuple)):
            # The StackedConvRNN2DCells isn't implemented yet.
            raise TypeError('It is not possible at the moment to'
                            'stack convolutional cells.')
        super(GraphConvRNN, self).__init__(cell,
                                           return_sequences,
                                           return_state,
                                           go_backwards,
                                           stateful,
                                           unroll,
                                           **kwargs)
        self.input_spec = [InputSpec(ndim=4)]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        cell = self.cell

        # Given:
        # - N: number of nodes in the graph
        # - F: filters or feature maps
        # Then:
        # - Z (`conv_out`) ∈ R^(N×F): the convolved signal matrix
        if self.cell.data_format == 'channels_first':
            num_nodes = input_shape[-1]
        elif self.cell.data_format == 'channels_last':
            num_nodes = input_shape[-2]

        output_shape = input_shape[:2] + (num_nodes, cell.filters)
        output_shape = transpose_shape(output_shape, cell.data_format,
                                       spatial_axes=[2])

        if not self.return_sequences:
            output_shape = output_shape[:1] + output_shape[2:]

        if self.return_state:
            output_shape = [output_shape]
            base = (input_shape[0], num_nodes, cell.filters)
            base = transpose_shape(base, cell.data_format, spatial_axes=[2])
            output_shape += [base[:] for _ in range(2)]
        return output_shape

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(
            shape=(batch_size, None) + input_shape[-2:])

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if self.cell.data_format == 'channels_first':
                ch_dim = -2
            elif self.cell.data_format == 'channels_last':
                ch_dim = -1
            if not [spec.shape[ch_dim]
                    for spec in self.state_spec] == state_size:
                raise ValueError(
                    'An initial_state was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'However `cell.state_size` is '
                    '{}'.format([spec.shape for spec in self.state_spec],
                                self.cell.state_size))
        else:
            if self.cell.data_format == 'channels_first':
                self.state_spec = [InputSpec(shape=(None, dim, None, None))
                                   for dim in state_size]
            elif self.cell.data_format == 'channels_last':
                self.state_spec = [InputSpec(shape=(None, None, None, dim))
                                   for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # (samples, timesteps, nodes, filters)
        initial_state = K.zeros_like(inputs)
        # (samples, nodes, filters)
        initial_state = K.sum(initial_state, axis=1)
        shape = list(self.cell.kernel_shape)
        shape[-1] = self.cell.filters

        if K.backend() == 'tensorflow':
            # We need to force this to be a tensor
            # and not a variable, to avoid variable initialization
            # issues.
            import tensorflow as tf
            kernel = tf.zeros(tuple(shape))
        else:
            kernel = K.zeros(tuple(shape))
        initial_state = self.cell.input_conv(initial_state,
                                             kernel)

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = _standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(GraphConvRNN, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = []
            for state in initial_state:
                try:
                    shape = K.int_shape(state)
                # Fix for Theano
                except TypeError:
                    shape = tuple(None for _ in range(K.ndim(state)))
                self.state_spec.append(InputSpec(shape=shape))

            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != K.is_keras_tensor(
                    additional_inputs[0]):
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors')

        if K.is_keras_tensor(additional_inputs[0]):
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(GraphConvRNN, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(GraphConvRNN, self).__call__(inputs, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        timesteps = K.int_shape(inputs)[1]

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True

        if self.return_state:
            states = to_list(states, allow_tuple=True)
            return [output] + states
        else:
            return output

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        input_shape = self.input_spec[0].shape
        state_shape = self.compute_output_shape(input_shape)
        if self.return_state:
            state_shape = state_shape[0]
        if self.return_sequences:
            state_shape = state_shape[:1] + state_shape[2:]
        if None in state_shape:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.\n'
                             'The same thing goes for the number of rows '
                             'and columns.')

        # helper function
        def get_tuple_shape(nb_channels):
            result = list(state_shape)
            if self.cell.data_format == 'channels_first':
                result[1] = nb_channels
            elif self.cell.data_format == 'channels_last':
                result[3] = nb_channels
            else:
                raise KeyError
            return tuple(result)

        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros(get_tuple_shape(dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros(get_tuple_shape(dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros(get_tuple_shape(self.cell.state_size)))
        else:
            states = to_list(states, allow_tuple=True)
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != get_tuple_shape(dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str(get_tuple_shape(dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
                K.set_value(state, value)

    def get_config(self):
        config = {'filters': self.filters,
                  'graph': self.graph_data,
                  'data_format': self.data_format,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(
                      self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(GraphConvRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GraphConvLSTM(GraphConvRNN):

    # TODO: check what is this annotation doing:
    # @interfaces.legacy_convlstm2d_support
    def __init__(self,
                 graph_data,
                 filters,
                 data_format=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = GraphConvLSTMCell(
            graph_data=graph_data,
            filters=filters,
            data_format=data_format,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(GraphConvLSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(GraphConvLSTM, self).call(inputs,
                                               mask=mask,
                                               training=training,
                                               initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def graph(self):
        return self.cell.graph

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'graph': self.graph_data,
                  'data_format': self.data_format,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(
                      self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(GraphConvLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GraphConvLSTMCell(Layer):

    def __init__(self,
                 graph_data,
                 filters,
                 data_format=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(GraphConvLSTMCell, self).__init__(**kwargs)

        if graph_data is None:
            raise ValueError('The graph data must be defined. Found `None`.')

        self.graph_data = graph_data
        self.filters = filters
        self.data_format = K.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self._conv_part = None

    def build(self, input_shape):
        # Build graph:
        # XXX: 2020-05-17: continue here - there is a problem using 
        # 'nx.Graph' type. Keras/Tensorflow seems not to be compatible with it.
        self.graph = Graph(self.graph_data)

        # Build kernels:
        if self.data_format == 'channels_first':
            channel_axis = -2
        elif self.data_format == 'channels_last':
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = (input_dim, self.filters * 4)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = (self.filters, self.filters * 4)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                @K.eager
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters, ), *args,
                                              **kwargs),
                        initializers.Ones()((self.filters, ), *args, **kwargs),
                        self.bias_initializer((self.filters * 2, ), *args,
                                              **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(
                shape=(self.filters * 4, ),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.filters]
        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.filters]
        self.kernel_f = self.kernel[:, self.filters:self.filters * 2]
        self.recurrent_kernel_f = (
            self.recurrent_kernel[:, self.filters:self.filters * 2])
        self.kernel_c = self.kernel[:, self.filters * 2:self.filters * 3]
        self.recurrent_kernel_c = (
            self.recurrent_kernel[:, self.filters * 2:self.filters * 3])
        self.kernel_o = self.kernel[:, self.filters * 3:]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.filters * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.filters]
            self.bias_f = self.bias[self.filters:self.filters * 2]
            self.bias_c = self.bias[self.filters * 2:self.filters * 3]
            self.bias_o = self.bias[self.filters * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None

        super(GraphConvLSTMCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[1]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        # TODO: review cell operations by formulas in paper:
        x_i = self.input_conv(inputs_i, self.kernel_i, self.bias_i)
        x_f = self.input_conv(inputs_f, self.kernel_f, self.bias_f)
        x_c = self.input_conv(inputs_c, self.kernel_c, self.bias_c)
        x_o = self.input_conv(inputs_o, self.kernel_o, self.bias_o)
        h_i = self.recurrent_conv(h_tm1_i, self.recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, self.recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, self.recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, self.recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h, c]

    def input_conv(self, x, w, b=None):
        conv_out = self.graph_conv(x, w,
                                   self.graph,
                                   self.data_format)

        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        return self.graph_conv(x, w,
                               self.graph,
                               self.data_format)

    # TODO: move this to K (Tensorflow backend)
    def graph_conv(self, x, kernel, graph, data_format=None):
        """Graph convolution, as per 'Kipf and Welling'.

        # Arguments
            x: Tensor or variable.
            kernel: kernel tensor.
            adj: adjacency matrix tensor.
            data_format: string, `"channels_last"` or `"channels_first"`.

        Given:
        - N (`num_nodes`): number of nodes in the graph
        - X (`x`) ∈ R^(N×C): input signal
        - C (`num_channels`): input channels (i.e. a C-dimensional 
                feature vector for every node)
        - F (`num_filters`): filters or feature maps
        - Θ (`w`) ∈ R^(C×F): a matrix of filter parameters
        - Z (`conv_out`) ∈ R^(N×F): the convolved signal matrix

        Then:
        `x`        ~> [N, C] ~> [27,  3]
        `w`        ~> [C, F] ~> [ 3, 10]
        `conv_out` ~> [N, F] ~> [27, 10]
        """
        num_channels, num_filters = K.int_shape(kernel)
        num_nodes = len(graph)
        assert K.int_shape(x)[-2:] == (num_nodes, num_channels)
        assert K.int_shape(kernel) == (num_channels, num_filters)

        if True: #self._conv_part is None:
            # FIXME: stop parsing sparse matrix to dense (Keras bug):
            adj = adjacency_matrix(graph).todense()
            adj = K.variable(adj)

            # Preprocessing from 'Kipf and Welling':
            i = K.eye(K.int_shape(adj)[0])
            a_hat = adj + i

            # TODO: review degree:
            # FIXME: stop parsing sparse matrix to dense (Keras bug):
            d_hat = K.variable(utils.degree(graph).todense())

            d_hat_inv_sqrt = K.pow(d_hat, -0.5)
            self._conv_part = d_hat_inv_sqrt * a_hat * d_hat_inv_sqrt  # (N, N)

        # Convolution:
        w_l = K.expand_dims(kernel, 0)
        x_theta = K.batch_dot(x, w_l)

        z_l = K.expand_dims(self._conv_part, 0)
        z = K.batch_dot(z_l, x_theta)

        assert K.int_shape(z)[-2:] == (num_nodes, num_filters)
        return z

    def get_config(self):
        config = {'filters': self.filters,
                  'graph': self.graph_data,
                  'data_format': self.data_format,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(
                      self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(GraphConvLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
