"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/stable/guides.html#readers
"""
import numpy as np
import nibabel as ni

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".nii.gz"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # # handle both a string and a list of strings
    # paths = [path] if isinstance(path, str) else path
    # # load all files into array
    # arrays = [ni.load(_path).get_fdata() for _path in paths]
    # # stack arrays into single array
    # data = np.squeeze(np.stack(arrays))

    # # optional kwargs for the corresponding viewer.add_* method
    # add_kwargs = {}
    # print(str(data.dtype))
    # if 'int' in str(data.dtype) :
    #     layer_type='labels'
    # else:
    #     layer_type = "image"  # optional, default is "image"
    # return [(data, add_kwargs, layer_type)]
        # handle both a string and a list of strings
    
    # load all files into array
    file = ni.load(path)
    # stack arrays into single array
    data = file.get_fdata()

    add_kwargs={"name":str(path.split('/')[-1]).replace('nii.gz',''), 'metadata':dict(affine=file.affine, header=file.header)}
    print('coucou',file.get_data_dtype())
    if 'uint' in str(file.get_data_dtype()) :
        layer_type='labels'
        data=data.astype('uint8')
    else:
        layer_type = "image" 
        data=data.astype('float32') # optional, default is "image"
    return [(data, add_kwargs, layer_type)]