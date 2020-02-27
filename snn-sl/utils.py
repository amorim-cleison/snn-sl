import pickle as pkl
import numpy as np
from tensorflow.keras import utils

def load_data(data_path, label_path, mmap=True, truncate_items=None):
    """
    Load data and label from the path.

    Parameters
    ----------
    data_path : str
        Path to data file, with .npy extension.
    label_path : str
        Path to file containing labels, with .pkl extension.
    mmap : bool, optional
        Wheter or not to use memory-mapped array representation.
    truncate_items : number, optional
        Number of items to truncate loaded data.

    Returns
    ------
    Dictionary with the attributes:
    - `data` : numpy array
        Data with the shape (N, C, T, V, M)
    - `labels` : list
        Labels of the data
    - `sample_names` : list
        Name of the corresponding .json files for the samples in data
    - `N` : int
        Denotes the batch size
    - `C` : int
        Denotes the coordinate dimensions of joints
    - `T` : int
        Denotes the length of frames
    - `V` : int
        Denotes the number of joints each frame
    - `M` : int 
        Denotes the number of people in the scene
    """
    # Load label:
    with open(label_path, 'rb') as f:
        sample_names, labels = pkl.load(f)

    # Load data:
    if mmap:
        data = np.load(data_path, mmap_mode='r')
    else:
        data = np.load(data_path)

    # Truncate data length to 'truncate_items', whether defined:
    if truncate_items:
        labels = labels[0:truncate_items]
        data = data[0:truncate_items]
        sample_names = sample_names[0:truncate_items]

    return {
        'data': data,
        'labels': labels,
        'sample_names': sample_names,
        'N': data.shape[0],
        'C': data.shape[1],
        'T': data.shape[2],
        'V': data.shape[3],
        'M': data.shape[4]
    }


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
        for T in N]      # 'timestep' in batch (if not zero)
            # if np.count_nonzero(T) > 0]
        for N in data])  # 'batch' in data


def map_to_flat(data):
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
        for T in N]             # 'timestep' in batch (if not zero)
            # if np.count_nonzero(T) > 0]
        for N in data])         # 'batch' in data


def prepare_labels(labels, num_classes):
    return np.asarray(utils.to_categorical(labels, num_classes=num_classes))