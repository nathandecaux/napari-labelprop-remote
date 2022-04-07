
from skimage import data
import nibabel as ni
import napari
import numpy as np
from kornia.geometry.transform import warp_perspective
import torch
# viewer = napari.view_path('/home/nathan/PLEX/norm/sub-002/img.nii.gz')
img=ni.load('img.nii.gz').get_fdata()
affine=torch.eye(3)
test=img[:,:,40].astype('float32')
test=warp_perspective(torch.from_numpy(test[None,None]),affine[None],test.shape).numpy()
viewer = napari.view_image(test)#affine=affine,blending='additive')
# viewer.add_image(img[:,:,50:51],affine=affine2,blending='additive')

napari.run()