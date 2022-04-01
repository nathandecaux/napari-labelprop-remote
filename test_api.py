from tkinter import Y
import requests
import json
from enum import Enum
import nibabel as ni
import io
import numpy as np
import base64
from urllib.parse import urljoin
import zipfile
import time
# from urllib.request import urlopen,urlretrieve
import cgi
host="10.29.225.156"
port="5000"
url = f'http://{host}:{port}'

def create_buf_npz(array_dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, **array_dict)
    buf.seek(0)
    return buf

def get_ckpts():
    r=urljoin(url,'list_ckpts')
    response = requests.get(r).text
    return response.split(',')

def get_file(url):
    """
    Download file without saving from url and return the bytes
    """
    return io.BytesIO(requests.get(url).content)

def get_session_info(token):
    r=urljoin(url,'get_session_info')
    response = requests.get(r,params={'token':token}).text
    return json.loads(response)

def session_exists(token):
    r=urljoin(url,'get_session_list')
    response = requests.get(r).text
    print(response)
    if token in response:
        return True
    else:
        return False



# buf = io.BytesIO()

# img=ni.load('img.nii.gz').get_fdata().astype('float32')
# mask=ni.load('mask.nii.gz').get_fdata().astype('uint8')
# np.savez_compressed(buf, img=img,mask=mask)
# buf.seek(0)
# params={'z_axis':2,'label':0,'checkpoint':'test.ckpt'}

# # Closed the buffer
# params=json.dumps(params).encode('utf-8')
# r=urljoin(url,'inference')
# #Send post request to http://10.29.225.156:5000/inference
# response=requests.post(r,files={'arrays':buf,'params':params})
# token=response.text
# buf.close()
token="7938470b-30bb-4401-8235-00cbbd092eba"
print(token)
#Send get request to http://10.29.225.156:5000/download_inference
while not session_exists(token):
    time.sleep(5)
print(get_session_info(token))

r=urljoin(url,'download_inference')
# response=requests.get(r,params={'token':token})
r=r+'?token='+token
# response=request.urlretrieve(r,'/tmp/test.npz')

#Recover arrays
# print(response.headers.get('content-type'))
# print(requests.get(urljoin(url,'get_session_list')).text)
npz_file=np.load(get_file(r),encoding = 'latin1')
print(np.unique(npz_file['Y_fused']))

