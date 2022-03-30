import requests
import json
from enum import Enum
import nibabel as ni
import io
import numpy as np
import base64

def get_ckpts():
    url = 'http://10.29.225.156:5000/list_ckpts'
    response = requests.get(url).text
    return response.split(',')

buf = io.BytesIO()

img=ni.load('/mnt/freebox/Segmentations/sub-17/ses-01/anat/sub-17_ses-1_DIXON6ECHOS-e3.nii.gz').get_fdata().astype('float32')
mask=ni.load('/mnt/freebox/Segmentations/sub-17/ses-01/anat/seg.nii.gz').get_fdata().astype('uint8')
np.savez_compressed(buf, img=img,mask=mask)
buf.seek(0)
params={'z_axis':2,'label':0,'checkpoint':'test.ckpt'}

# Closed the buffer
params=json.dumps(params).encode('utf-8')
url="http://10.29.225.156:5000/inference"
#Send post request to http://10.29.225.156:5000/inference
response=requests.post(url,files={'arrays':buf,'params':params})
print(response.text)
buf.close()
