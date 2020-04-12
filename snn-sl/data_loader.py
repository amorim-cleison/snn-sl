import pickle as pkl
from os.path import isdir

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def __load_data_and_label(data_path, label_path, mmap=True, truncate_items=None):
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


def load_data(path: str):
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
    
    flatten_data : bool
        Indicates when data must be flattened to 3d.

    Returns
    ----------
    A list of the following data, in this sequence:

    - X_train : numpy array
        Training data.
    - y_train : numpy array
        Labels for the training data.
    - X_test : numpy array
        Test / validation data.
    - y_test : numpy array
        Labels for the test / validation data.
    - num_classes : integer
        Number of classes (or categories) in loaded data.
    """
    if not isdir(path):
        raise FileNotFoundError("Invalid path: '{0}'.", path)

    data_path = '{0}/{1}_data.npy'
    label_path = '{0}/{1}_label.pkl'

    train = __load_data_and_label(
        data_path.format(path, 'train'), label_path.format(path, 'train'))
    test = __load_data_and_label(
        data_path.format(path, 'test'), label_path.format(path, 'test'))

    X_train = train['data']
    y_train = train['labels']
    X_test = test['data']
    y_test = test['labels']
    num_classes = len(np.unique(y_train))

    return (X_train, y_train, X_test, y_test, num_classes)


def prepare_data_and_label(X_train, y_train, X_test, y_test):
    return (__prepare_data(X_train), __prepare_label(y_train),
            __prepare_data(X_test), __prepare_label(y_test))


def __prepare_data(data):
    """
    Apply data transformations
    """
    new_data = data

    #------------------------------------------------------------
    # PREPARATION FOR 'LSTM':
    #
    # # Transform data from shape
    # # (N, C, T, V, M) / (87, 3, 60, 27, 1) to
    # # (N, T, M, V, C) / (87, 60, 1, 27, 3),
    # # by moving axes [1, 2, 4] to [4, 1, 2]
    # new_data = np.moveaxis(new_data, [1, 2, 4], [4, 1, 2])

    # # Remove single-dimensional entries from the shape of data,
    # # transforming shape to (N, T, V, C) or (87, 60, 27, 3):
    # new_data = np.squeeze(new_data)

    # # Remove empty timestemps:
    # # new_data = remove_empty_timesteps(new_data)

    # # Map 3rd dimension to a dict:
    # new_data = __map_to_flat(new_data)
    #------------------------------------------------------------

    #------------------------------------------------------------
    # PREPARATION FOR 'ConvLSTM':
    # # (N, C, T, V, M) / (87, 3, 60, 27, 1) to
    # # (N, T, M, V, C) / (87, 60, 1, 27, 3),
    # new_data = np.moveaxis(new_data, [1, 2, 4], [4, 1, 2])
    #------------------------------------------------------------

    #------------------------------------------------------------
    # PREPARATION FOR 'AGC LSTM':
    # (N, C, T, V, M) / (87, 3, 60, 27, 1) to
    # (N, T, M, V, C) / (87, 60, 1, 27, 3),
    # new_data = np.moveaxis(new_data, [1, 2, 4], [4, 1, 2])
    # new_data = np.squeeze(new_data)
    #------------------------------------------------------------

    return new_data


def __prepare_label(label):
    return np.asarray(label)


def __map_to_json(data):
    """
    Map to a JSON representation with the following layout:
    `{ 'x': 0.0, 'y': 0.0, 'precision': 0.0 }`
    """
    # (N, T, V, C) or (87, 60, 27, 3)
    return np.asarray([
        [
            [{
                'x': V[0],
                'y': V[1],
                'precision': V[2]
            } for V in T]  # 'joint' in timestep
            for T in N
        ]  # 'timestep' in batch
        for N in data
    ])  # 'batch' in data


def __map_to_flat_with_precision(data):
    """
    Map to a flat representation, where the coordinates X, Y, 
    and Precision are concatenated in this sequence.
    """
    # Transpose: rotate X, Y, Precision
    # Ravel: flatten all X coordinates, followed by Y coordinates
    #       and Precision information
    # (N, T, V, C) or (87, 60, 27, 3)
    return np.asarray([
        [T.transpose().ravel() for T in N]  # 'timestep' in batch
        for N in data
    ])  # 'batch' in data


def __map_to_flat(data):
    """
    Map to a flat representation, where the coordinates X, Y, 
    and Precision are concatenated in this sequence.
    """
    # Transpose: rotate X, Y, Precision
    # Ravel: flatten all X coordinates, followed by Y coordinates
    #       and Precision information
    # (N, T, V, C) or (87, 60, 27, 3)
    # Indexes 0 and 1 correspond to X and Y
    return np.asarray([
        [T.transpose()[:2].ravel() for T in N]  # 'timestep' in batch
        for N in data
    ])  # 'batch' in data


def __remove_empty_timesteps(data):
    """
    Remove empty frames from data
    """
    return np.asarray([
        [T for T in N if np.count_nonzero(T) > 0]  # 'timestep' in batch
        for N in data
    ])  # 'batch' in data
