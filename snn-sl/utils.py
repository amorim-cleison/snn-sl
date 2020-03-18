import pickle as pkl

import numpy as np


def load_data_and_label(data_path, label_path, mmap=True, truncate_items=None):
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
