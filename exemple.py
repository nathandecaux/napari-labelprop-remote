
from random import Random
from skimage import data
import nibabel as ni
import napari
import numpy as np
from kornia.geometry.transform import warp_perspective
from kornia.augmentation.augmentation import RandomAffine,RandomHorizontalFlip,RandomVerticalFlip
from kornia.augmentation.container import AugmentationSequential
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import conv2d
import math
from torch.nn import functional as F
import functools
import hashlib
import time

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred, mean=True):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win])

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        if mean:
            return -torch.mean(cc)
        else:
            return cc


def timer(func):
    @functools.wraps(func) #optional line if you went the name of the function to be maintained, to be imported
    def wrapper(*args, **kwargs):
        start = time.time()
        #do somehting with the function
        value = func(*args, **kwargs)
        end = time.time()
        print(end-start)
        return value
    return wrapper

@timer
def hash_array(array):
    return hashlib.md5(array.tobytes()).hexdigest()
# viewer = napari.view_path('IRM_dA_but_20180919150646_301.nii.gz')
# viewer.open('sparse_seg.nii.gz')
img=ni.load('img.nii.gz').get_fdata()
mask=ni.load('mask.nii.gz').get_fdata()
print(img.shape)
print(hash_array(img))
# trans=RandomAffine([3.,6.],[0.,1e-3],p=1)
# trans=AugmentationSequential(RandomVerticalFlip(p=0),RandomHorizontalFlip(p=0))

# im=torch.from_numpy(img[...,40].astype('float32'))[None,None]
# gt=torch.from_numpy(mask[...,40])[None,None]
# # affine=torch.eye(3)
# # test=img[:,:,40].astype('float32')
# # test=warp_perspective(torch.from_numpy(test[None,None]),affine[None],test.shape).numpy()
# # viewer = napari.view_image(test)#affine=affine,blending='additive')
# # viewer.add_image(img[:,:,50:51],affine=affine2,blending='additive')
# plt.figure()
# # plt.imshow(NCC().loss(trans(im),im,False)[0,0])
# plt.imshow((im-trans(im))[0,0])
# plt.show()
# napari.run()