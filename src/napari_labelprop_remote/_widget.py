"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QHBoxLayout
from magicgui import magic_factory,magicgui
from magicgui.widgets import Select,Slider,PushButton,FileEdit,Container,Label,LineEdit
from magicgui.widgets import FunctionGui

import requests
from enum import Enum,unique
import requests
import json
from enum import Enum
import nibabel as ni
import io
import numpy as np
from urllib.parse import urljoin
import time
import functools
import hashlib
from napari.plugins import NapariPluginManager
import napari
import os
#Get path to this package
package_path = os.path.dirname(os.path.abspath(__file__))
print(package_path)
global server
global url
server=json.load(fp=open(os.path.join(package_path,'conf.json')))
url = f'http://{server["host"]}:{server["port"]}'

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
def create_buf_npz(array_dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, **array_dict)
    buf.seek(0)
    return buf

def get_url():
    server=json.load(fp=open(os.path.join(package_path,'conf.json')))
    return f'http://{server["host"]}:{server["port"]}'

def get_ckpts(host='configured'):
    if host=='configured':
        host=get_url()
    try :

        r=urljoin(host,'list_ckpts')
        #send request (3 retries max)
        response = requests.get(r,timeout=3).text
    except :
        response = 'Server Unavailable'
        pass
    finally:
        ckpts=response.split(',')
        ckpts.sort()
        return ckpts


@timer
def hash_array(array):
    return hashlib.md5(array.tobytes()).hexdigest()

def get_hash():
    try :
        r=urljoin(get_url(),'list_hash')
        #send request (3 retries max)
        response = requests.get(r,timeout=3).text
    except :
        response = 'Server Unavailable'
        pass
    finally:
        hash_list=response.split(',')
        return hash_list

def get_file(url):
    """
    Download file without saving from url and return the bytes
    """
    return requests.get(url)#io.BytesIO(requests.get(url).content)

def get_session_info(token):
    r=urljoin(get_url(),'get_session_info')
    response = requests.get(r,params={'token':token}).text
    return json.loads(response)

def session_exists(token):
    r=urljoin(get_url(),'get_session_list')
    response = requests.get(r).text
    if token in response:
        return True
    else:
        return False

def send_ckpt(ckpt : str) -> bool:
    """Send file to server"""

    r=urljoin(get_url(),'send_ckpt')
    response = requests.post(r,files={'ckpt':open(ckpt)}).text
    if response=='ok':
        return True
    else:
        return False

def get_ckpt_dir():
    """Send request get_ckpt_dir to server"""
    r=urljoin(get_url(),'get_ckpt_dir')
    try:
        response = requests.get(r,timeout=3).text
    except:
        response = 'Server Unavailable'
        pass
    return response

def set_ckpt_dir(ckpt_dir):
    r=urljoin(get_url(),'set_ckpt_dir')
    response = requests.post(r,params={'ckpt_dir':ckpt_dir}).text
    return response

def configure_server(host : str=server["host"],port : str=server["port"]) -> None :
    list_ckpts=get_ckpts(f'http://{host}:{port}')
    if 'Server Unavailable' in list_ckpts:
        napari.utils.notifications.show_error('Server Unavailable')
        return False
    else:
        server["host"]=host
        server["port"]=port
        json.dump(server,fp=open(os.path.join(package_path,'conf.json'),'w'))
        napari.utils.notifications.show_info('Server configured and connected')
        return True



class settings(FunctionGui):
    def __init__(self):
        super().__init__(configure_server,call_button = 'Configure Server')
        
        self.status=Label()
        list_ckpts=get_ckpts()
        self.insert(2,self.status)
        self.checkpoint_dir=LineEdit(label='checkpoint directory')
        self.ckpt_dir_btn=PushButton(label='Set checkpoint directory')
        self.ckpt_dir_btn.clicked.connect(self._set_ckpt_dir)
        self.insert(-1,self.checkpoint_dir)
        self.insert(-1,self.ckpt_dir_btn)
        self.send_button=PushButton(text='Send checkpoint')
        self.send_button.clicked.connect(self._send_ckpt)
        self.ckpt=FileEdit(label='Send checkpoint to server')
        self.insert(-1,self.ckpt)
        self.insert(-1,self.send_button)

        if 'Server Unavailable' in list_ckpts:
            self.status.value='Server Unavailable'
            self.checkpoint_dir.hide()
            self.ckpt_dir_btn.hide()
            self.send_button.hide()
            self.ckpt.hide()
        else:
            self.status.value='Server Connected'
            self.checkpoint_dir.value=get_ckpt_dir()



    def __call__(self):
        server_ok=super().__call__()
        if server_ok:
            self.status.value='Server Connected'
            self.checkpoint_dir.show()
            self.ckpt_dir_btn.show()
            self.send_button.show()
            self.ckpt.show()
            self.checkpoint_dir.value=get_ckpt_dir()
        else:
            self.status.value='Server Unavailable'
            self.checkpoint_dir.hide()
            self.ckpt_dir_btn.hide()
            self.send_button.hide()
            self.ckpt.hide()
    def _send_ckpt(self):
        ckpt=self.ckpt.value
        if send_ckpt(ckpt):
            napari.utils.notifications.show_info('Checkpoint sent')
        else:
            napari.utils.notifications.show_error('Checkpoint not sent')
    def _set_ckpt_dir(self):
        ckpt_dir=self.checkpoint_dir.value
        set_ckpt_dir(ckpt_dir)
        napari.utils.notifications.show_info('Checkpoint directory set')





        
               

# @magic_factory(checkpoint={'choices':['']+get_ckpts()},criteria={'choices':['distance','ncc']},reduction={'choices':['none','local_mean','mean']})
def inference_function(image: "napari.layers.Image", labels: "napari.layers.Labels",z_axis: int, label : int, checkpoint:"napari.types.Path",criteria='ncc',reduction='none',gpu=True) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    r=urljoin(get_url(),'inference')
    if gpu:
        device='cuda'
    else:
        device='cpu'
    params={'z_axis':z_axis,'label':label,'checkpoint':checkpoint,'criteria':criteria,'reduction':reduction,'device':device}
    hash=hash_array(image.data.astype('float32'))
    list_hash=get_hash()
    if hash in list_hash:
        params['hash']=hash
        buf=create_buf_npz({'mask':labels.data.astype('uint8')})
    else:
        buf=create_buf_npz({'img':image.data.astype('float32'),'mask':labels.data.astype('uint8')})
    
    params=json.dumps(params).encode('utf-8')
    response=requests.post(r,files={'arrays':buf,'params':params})
    token=response.text
    buf.close()
    while not session_exists(token):
        time.sleep(5)
    r=urljoin(get_url(),'download_inference')
    r=r+'?token='+token
    response=get_file(r)
    try:
        npz_file=np.load(io.BytesIO(response.content),encoding = 'latin1')
        Y_up,Y_down,Y_fused=npz_file['Y_up'],npz_file['Y_down'],npz_file['Y_fused']
        return [((Y_up).astype('uint8'), {'name': 'propagated_up','metadata':labels.metadata}, 'labels'), ((Y_down).astype('uint8'), {'name': 'propagated_down','metadata':labels.metadata}, 'labels'), ((Y_fused).astype('uint8'), {'name': 'propagated_fused','metadata':labels.metadata}, 'labels')]
    except:
        #Convert f as string
        e=response.text
        print(e)
        #Raise exception with f as the message
        raise Exception('Server-side error: '+e)
    # shape=torch.load(checkpoint)['hyper_parameters']['shape'][0]
    # if label==0: label='all'
    # Y_up, Y_down, Y_fused = propagate_from_ckpt(
    #     image, labels, checkpoint, z_axis=z_axis,lab=label,shape=shape)
    # return [((Y_up).astype(int), {'name': 'propagated_up'}, 'labels'), ((Y_down).astype(int), {'name': 'propagated_down'}, 'labels'), ((Y_fused).astype(int), {'name': 'propagated_fused'}, 'labels')]

class inference(FunctionGui):
    def __init__(self):
        super().__init__(inference_function,call_button=True,param_options={'checkpoint':{'choices':['']+get_ckpts()},'criteria':{'choices':['distance','ncc']},'reduction':{'choices':['none','local_mean','mean']}})
        refresh_btn=PushButton()
        # file_select=FileEdit()
        # file_select.label='or select locally'
        # file_select.choices=get_ckpts()
        refresh_btn.clicked.connect(self._on_click)
        refresh_btn.text='Refresh'
        # self.insert(5,refresh_btn)
        # self.insert(6,file_select)
        # container=Container(layout='horizontal',widgets=[refresh_btn,file_select])
        self.insert(5,refresh_btn)

    def __call__(self):
        super().__call__()
    def _on_click(self):
        self.checkpoint.choices = ['']+get_ckpts()

def training_function(image: "napari.layers.Image", labels: "napari.layers.Labels", pretrained_checkpoint: "napari.types.Path" = '', shape: int=256, z_axis: int=0, max_epochs: int=10,checkpoint_name='',criteria='ncc',reduction='none',gpu=True) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    r=urljoin(get_url(),'training')
    if gpu:
        device='cuda'
    else:
        device='cpu'
    params={'pretrained_ckpt':pretrained_checkpoint,'shape':shape,'z_axis':z_axis,'max_epochs':max_epochs,'name':checkpoint_name,'pretraining':False,'criteria':criteria,'reduction':reduction,'device':device}
    hash=hash_array(image.data.astype('float32'))
    list_hash=get_hash()
    if hash in list_hash:
        params['hash']=hash
        buf=create_buf_npz({'mask':labels.data.astype('uint8')})
    else:
        buf=create_buf_npz({'img':image.data.astype('float32'),'mask':labels.data.astype('uint8')})
    
    params=json.dumps(params).encode('utf-8')

    response=requests.post(r,files={'arrays':buf,'params':params})
    token=response.text
    buf.close()
    while not session_exists(token):
        time.sleep(5)
    r=urljoin(get_url(),'download_inference')
    r=r+'?token='+token
    response=get_file(r)
    try:
        npz_file=np.load(io.BytesIO(response.content),encoding = 'latin1')
        Y_up,Y_down,Y_fused=npz_file['Y_up'],npz_file['Y_down'],npz_file['Y_fused']
        return [((Y_up).astype('uint8'), {'name': 'propagated_up','metadata':labels.metadata}, 'labels'), ((Y_down).astype('uint8'), {'name': 'propagated_down','metadata':labels.metadata}, 'labels'), ((Y_fused).astype('uint8'), {'name': 'propagated_fused','metadata':labels.metadata}, 'labels')]
    except:
        #Convert f as string
        e=response.text
        print(e)
        #Raise exception with f as the message
        raise Exception('Server-side error: '+e)

class training(FunctionGui):
    def __init__(self):
        super().__init__(training_function,call_button=True,param_options={'pretrained_checkpoint':{'choices':['']+get_ckpts()},'criteria':{'choices':['distance','ncc']},'reduction':{'choices':['none','local_mean','mean']}})
        btn=PushButton()
        btn.clicked.connect(self._on_click)
        btn.text='Refresh list'
        self.insert(5,btn)

    def __call__(self):
        super().__call__()
    def _on_click(self):
        self.pretrained_checkpoint.choices = ['']+get_ckpts()

#send get request to 10.29.225.156:5000/list_ckpts
#return list of ckpts

