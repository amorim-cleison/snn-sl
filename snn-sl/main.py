from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import utils as u
import nested_lstm as nlstm

"""
(87, 3, 60, 27, 1)
( N, C,  T,  V, M)
- N denotes the batch size
- C denotes the coordinate dimensions of joints
- T denotes the length of frames
- V denotes the number of joints each frame
- M denotes the number of people in the scene
"""
data_path = '../data/normalized/{0}'
num_classes = 2745

train = u.load_data(
            data_path.format('train_data.npy'), 
            data_path.format('train_label.pkl'))
test = u.load_data(
            data_path.format('test_data.npy'), 
            data_path.format('test_label.pkl'))

train_x = u.prepare_data(train['data'])
train_y = u.prepare_labels(train['labels'], num_classes)
test_x = u.prepare_data(test['data'])
test_y = u.prepare_labels(test['labels'], num_classes)

# train_x = np.random.random([batch_size, 60, 27]).astype(np.float32)
# train_y = np.random.random_integers(0, num_classes, [batch_size, 1])
# test_x  = np.random.random([batch_size, 60, 27]).astype(np.float32)
# test_y  = np.random.random_integers(0, num_classes, [batch_size, 1])


# ------------------- Architecture: -----------------------

# Input shape: 
# (N, T, V, C)
# (batch, timesteps, { joints, dimensions })
# (87, 60, { 27, 3 })

model = tf.keras.Sequential()

# model.add(layers.Embedding(input_dim=num_classes, output_dim=2745))

# Recurrent layer
# model.add(layers.LSTM(64, return_sequences=False, 
#                dropout=0.1, recurrent_dropout=0.1))
model.add(layers.LSTM(64))

# model.add(layers.RNN(nlstm.NestedLSTMCell(64, depth=2, input_shape=(60, 27, 3))))

# Fully connected layer
# model.add(layers.Dense(64, activation='relu'))

# Regularization:
# model.add(layers.Dropout(0.5))

# Output layer:
model.add(layers.Dense(num_classes, activation='softmax'))
# ---------------------------------------------------------


model.compile(
    loss='categorical_crossentropy', 
    # loss='sparse_categorical_crossentropy', 
    optimizer='adam', 
    metrics=['categorical_accuracy']
)
# model.summary()
# checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

# Train:
model.fit(
    x=train_x,
    y=train_y,
    epochs=100,
    batch_size=8)

# Evaluate:
result = model.predict(test_x, batch_size=8, verbose=0)
for value in result:
	print('%.1f' % value)

"""
model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))
"""

"""
model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()
"""

