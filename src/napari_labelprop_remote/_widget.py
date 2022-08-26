"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QHBoxLayout
from magicgui import magic_factory,magicgui
from magicgui.widgets import Select,Slider,PushButton,FileEdit,Container
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

class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"): 
    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.



def inject_items(d, items):
    for i,v in enumerate(items):
        d[str(v)] = v

class Checkpoints(Enum):
    inject_items(locals(), ['a','b','c'])

def configure_server(host : str=server["host"],port : str=server["port"]) -> None :
    list_ckpts=get_ckpts(f'http://{host}:{port}')
    if 'Server Unavailable' in list_ckpts:
        raise Exception('Server Unavailable')
    else:
        server["host"]=host
        server["port"]=port
        json.dump(server,fp=open(os.path.join(package_path,'conf.json'),'w'))
        print('Server configured and connected')
        napari.utils.notifications.show_info('Server configured and connected')



class settings(FunctionGui):
    def __init__(self):
        super().__init__(configure_server)

    def __call__(self):
        super().__call__()



        
               

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
        file_select=FileEdit()
        file_select.label='Select checkpoint from local file'
        file_select.choices=get_ckpts()
        refresh_btn.clicked.connect(self._on_click)
        refresh_btn.text='Refresh list'
        # self.insert(5,refresh_btn)
        # self.insert(6,file_select)
        container=Container(layout='horizontal',widgets=[refresh_btn,file_select])
        self.insert(5,container)

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

