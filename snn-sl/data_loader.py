from os.path import isdir

import numpy as np

import utils as u
import tensorflow.keras.utils as tfutils

def load_data(path: str, num_classes: int):
    """
    Load and prepare the data present in the presented path. 
    The input data is loaded in the layout:
    ```
    (87, 3, 60, 27, 1)
    ( N, C,  T,  V, M)
    ```
    and is transformed to the layout:
    ```
    ( N,  T,  K) 
    (87, 60, 81)
    ```
    Considering the following description for the dimensions above:
    - N : denotes the batch size
    - C : denotes the coordinate dimensions of joints
    - T : denotes the length of frames
    - V : denotes the number of joints each frame
    - M : denotes the number of people in the scene
    - K : composition of the C and V dimensions into a single flattened array

    Parameters
    ----------
    path : str
        The path to the data to be loaded.
        
    num_classes : int
        Number of classes (or categories) to consider while 
        preparing labels.

    Returns
    ----------
    A list of the following data, in this sequence:

    - train_x : numpy array
        Training data.
    - train_y : numpy array
        Labels for the training data.
    - train_samples : list
        Name of the JSON files for every item in the training labels.
    - test_x : numpy array
        Test / validation data.
    - test_y : numpy array
        Labels for the test / validation data.
    - test_samples : list
        Name of the JSON files for every item in the test / validation labels.
    """
    if not isdir(path):
        raise FileNotFoundError("Invalid path: '{0}'.", path)

    data_path = '{0}/{1}_data.npy'
    label_path = '{0}/{1}_label.pkl'

    train = u.load_data(
        data_path.format(path, 'train'), label_path.format(path, 'train'))
    test = u.load_data(
        data_path.format(path, 'test'), label_path.format(path, 'test'))

    train_x = prepare_data(train['data'])
    # train_y = prepare_labels(train['labels'], num_classes)
    train_y = train['labels']
    train_samples = train['sample_names']

    test_x = prepare_data(test['data'])
    test_y = prepare_labels(test['labels'], num_classes)
    test_samples = test['sample_names']

    return (train_x, train_y, train_samples, test_x, test_y, test_samples)



def prepare_data(data):
    """
    Apply data transformations
    """
    new_data = data

    # Transform data from shape 
    # (N, C, T, V, M) / (87, 3, 60, 27, 1) to
    # (N, T, M, V, C) / (87, 60, 1, 27, 3), 
    # by moving axes [1, 2, 4] to [4, 1, 2]
    new_data = np.moveaxis(new_data, [1, 2, 4], [4, 1, 2])

    # Remove single-dimensional entries from the shape of data,
    # transforming shape to (N, T, V, C) or (87, 60, 27, 3):
    new_data = np.squeeze(new_data)

    # Remove empty timestemps:
    # new_data = remove_empty_timesteps(new_data)

    # Map 3rd dimension to a dict:
    new_data = map_to_flat(new_data)

    return new_data


def map_to_json(data):
    """
    Map to a JSON representation with the following layout:
    `{ 'x': 0.0, 'y': 0.0, 'precision': 0.0 }`
    """
    # (N, T, V, C) or (87, 60, 27, 3)
    return np.asarray([[[ 
        { 
            'x': V[0], 
            'y': V[1], 
            'precision': V[2] 
        } 
        for V in T]      # 'joint' in timestep
        for T in N]      # 'timestep' in batch
        for N in data])  # 'batch' in data


def map_to_flat_with_precision(data):
    """
    Map to a flat representation, where the coordinates X, Y, 
    and Precision are concatenated in this sequence.
    """
    # Transpose: rotate X, Y, Precision 
    # Ravel: flatten all X coordinates, followed by Y coordinates 
    #       and Precision information
    # (N, T, V, C) or (87, 60, 27, 3)
    return np.asarray([[
        T.transpose().ravel()      
        for T in N]         # 'timestep' in batch
        for N in data])     # 'batch' in data


def map_to_flat(data):
    """
    Map to a flat representation, where the coordinates X, Y, 
    and Precision are concatenated in this sequence.
    """
    # Transpose: rotate X, Y, Precision 
    # Ravel: flatten all X coordinates, followed by Y coordinates 
    #       and Precision information
    # (N, T, V, C) or (87, 60, 27, 3)
    # Indexes 0 and 1 correspond to X and Y
    return np.asarray([[
        T.transpose()[:2].ravel()      
        for T in N]         # 'timestep' in batch
        for N in data])     # 'batch' in data


def remove_empty_timesteps(data):
    """
    Remove empty frames from data
    """
    return np.asarray([[ T 
        for T in N if np.count_nonzero(T) > 0]  # 'timestep' in batch
        for N in data])                         # 'batch' in data

def prepare_labels(labels, num_classes):
    return np.asarray(tfutils.to_categorical(labels, num_classes=num_classes))
