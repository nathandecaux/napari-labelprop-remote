"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QHBoxLayout
from magicgui import magic_factory,magicgui
from magicgui.widgets import Select,Slider,PushButton,FileEdit,Container,Label,LineEdit,RadioButtons
from magicgui.widgets import FunctionGui
import sys
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
import pandas as pd
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
        # raise Exception(response)
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
        hash_list=response.split(',')
        return hash_list
    except :
        response = 'Server Unavailable'
        raise Exception(response)


def get_file(url):
    """
    Download file without saving from url and return the bytes
    """
    try:
        response = requests.get(url,timeout=10)
        return response
    except:
        raise Exception('Server Unavailable')

def get_session_info(token):
    r=urljoin(get_url(),'get_session_info')
    try:
        response = requests.get(r,params={'token':token}).text
        return json.loads(response)
    except :
        response = 'Server Unavailable'
        raise Exception(response)

def session_exists(token):
    r=urljoin(get_url(),'get_session_list')
    try:
        response = requests.get(r,timeout=5).text
        if token in response:
            return True
        else:
            return False
    except:
        response = 'Server Unavailable'
        raise Exception(response)

        
def send_ckpt(ckpt : str) -> bool:
    """Send file to server"""

    r=urljoin(get_url(),'send_ckpt')
    try:
        response = requests.post(r,files={'ckpt':open(ckpt)}).text
        if response=='ok':
            return True
        else:
            return False
    except:
        response = 'Server Unavailable'
        raise Exception(response)

def get_ckpt_dir():
    """Send request get_ckpt_dir to server"""
    r=urljoin(get_url(),'get_ckpt_dir')
    try:
        response = requests.get(r,timeout=3).text
        return response
    except:
        response = 'Server Unavailable'
        raise Exception(response)

def set_ckpt_dir(ckpt_dir):
    r=urljoin(get_url(),'set_ckpt_dir')
    try:
        response = requests.post(r,params={'ckpt_dir':ckpt_dir}).text
        return response
    except:
        response = 'Server Unavailable'
        raise Exception(response)

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
def inference_function(image: "napari.layers.Image", labels: "napari.layers.Labels",hints:"napari.layers.Labels",z_axis: int, label : int, checkpoint:str="",criteria='ncc',reduction='none',gpu=True) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    r=urljoin(get_url(),'inference')
    if label>0:
        hints=(hints==label)*1
    if gpu:
        device='cuda'
    else:
        device='cpu'
    params={'z_axis':z_axis,'label':label,'checkpoint':checkpoint,'criteria':criteria,'reduction':reduction,'device':device}
    hash=hash_array(image.data.astype('float32'))
    list_hash=get_hash()
    napari.utils.notifications.show_info('Compressing images')
    arrays={'mask':labels.data.astype('uint8')}
    if hash in list_hash:
        params['hash']=hash
    else:
        arrays['img']=image.data.astype('float32')
    if hints!='':
        arrays['hints']=hints.data.astype('uint8')
    buf=create_buf_npz(arrays)
    params=json.dumps(params).encode('utf-8')
    napari.utils.notifications.show_info('Sending request')
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
        return [((Y_up).astype('uint8'), {'name': 'propagated_up','metadata':labels.metadata,'scale':labels.scale}, 'labels'), ((Y_down).astype('uint8'), {'name': 'propagated_down','metadata':labels.metadata,'scale':labels.scale}, 'labels'), ((Y_fused).astype('uint8'), {'name': 'propagated_fused','metadata':labels.metadata,'scale':labels.scale}, 'labels')]
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
    def __init__(self,viewer: "napari.viewer.Viewer"):
        super().__init__(inference_function,call_button=True,param_options={'hints':{'choices':['']+[x for x in viewer.layers if isinstance(x,napari.layers.Labels)]},'checkpoint':{'choices':['']+get_ckpts()},'criteria':{'choices':['distance','ncc']},'reduction':{'choices':['none','local_mean','mean']}})
        refresh_btn=PushButton()
        # file_select=FileEdit()
        # file_select.label='or select locally'
        # file_select.choices=get_ckpts()
        self.checkpoint.choices=['']+get_ckpts()
        refresh_btn.clicked.connect(self.click)
        refresh_btn.text='Refresh'
        self.criteria.changed.connect(self.update_reduction)

        # self.insert(5,refresh_btn)
        # self.insert(6,file_select)
        # container=Container(layout='horizontal',widgets=[refresh_btn,file_select])
        self.insert(5,refresh_btn)
        viewer.layers.events.inserted.connect(self.update_hints)
        viewer.layers.events.removed.connect(self.update_hints)
        self.viewer=viewer
        self.update_hints()


    def __call__(self):
        napari.utils.notifications.show_info('Inference started')
        super().__call__()
        napari.utils.notifications.show_info('Inference finished')
    def click(self):
        self.checkpoint.choices = ['']+get_ckpts()
    def update_reduction(self):
        if self.criteria.value=='distance':
            self.reduction.value='mean'
            self.reduction.hide()
        else:
            self.reduction.show()
    def update_hints(self):
        print(self.viewer.layers)
        self.hints.choices=['']+[x for x in self.viewer.layers if isinstance(x,napari.layers.Labels)]

def training_function(image: "napari.layers.Image", labels: "napari.layers.Labels",hints:"napari.layers.Labels", pretrained_checkpoint: "napari.types.Path" = '', shape: int=256, z_axis: int=0, max_epochs: int=10,checkpoint_name='',criteria='ncc',reduction='none',gpu=True) -> "napari.types.LayerDataTuple":
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
    arrays={'mask':labels.data.astype('uint8')}
    if label>0:
        hints=(hints==label)*1
    if hash in list_hash:
        params['hash']=hash
    else:
        arrays['img']=image.data.astype('float32')
    if hints!='':
        arrays['hints']=hints.data.astype('uint8')
    buf=create_buf_npz(arrays)
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
        return [((Y_up).astype('uint8'), {'name': 'propagated_up','metadata':labels.metadata,'scale':labels.scale}, 'labels'), ((Y_down).astype('uint8'), {'name': 'propagated_down','metadata':labels.metadata,'scale':labels.scale}, 'labels'), ((Y_fused).astype('uint8'), {'name': 'propagated_fused','metadata':labels.metadata,'scale':labels.scale}, 'labels')]
    except:
        #Convert f as string
        e=response.text
        print(e)
        #Raise exception with f as the message
        raise Exception('Server-side error: '+e)

class training(FunctionGui):
    def __init__(self,viewer: "napari.viewer.Viewer"):
        super().__init__(training_function,call_button=True,param_options={'hints':{'choices':['']+[x for x in viewer.layers if isinstance(x,napari.layers.Labels)]},'pretrained_checkpoint':{'choices':['']+get_ckpts()},'criteria':{'choices':['distance','ncc']},'reduction':{'choices':['none','local_mean','mean']}})
        self.criteria.changed.connect(self.update_reduction)

        btn=PushButton()
        btn.clicked.connect(self._on_click)
        btn.text='Refresh list'
        self.insert(5,btn)
        viewer.layers.events.inserted.connect(self.update_hints)
        viewer.layers.events.removed.connect(self.update_hints)
        self.viewer=viewer

    def __call__(self):
        napari.utils.notifications.show_info('Training started')
        super().__call__()
        napari.utils.notifications.show_info('Training finished')    
    
    def _on_click(self):
        self.pretrained_checkpoint.choices = ['']+get_ckpts()

    def update_reduction(self):
        if self.criteria.value=='distance':
            self.reduction.value='mean'
            self.reduction.hide()
        else:
            self.reduction.show()
    def update_hints(self):
        print(self.viewer.layers)
        self.hints.choices=['']+[x for x in self.viewer.layers if isinstance(x,napari.layers.Labels)]

def set_label_colormap_function(labels_layer : "napari.layers.Labels",table:pd.DataFrame):
    return labels_layer

class set_label_colormap(FunctionGui):
    def __init__(self,viewer: "napari.viewer.Viewer"):
        super().__init__(set_label_colormap_function,call_button=False)
        self.viewer=viewer
        self.colormap_file=FileEdit(label='Get colormap from file')
        self.insert(-1,self.colormap_file)
        self.label_names=RadioButtons(label='Label names',choices=[])
        self.file_button=PushButton(label='Load colormap')
        self.file_button.clicked.connect(self.update_colormap)
        self.insert(-1,self.file_button)
        self.insert(-1,self.label_names)

        # self.colormap_file.changed.connect(self.update_colormap)
        self.label_names.changed.connect(self.update_selected_label)

    def update_colormap(self):
        if self.colormap_file.value!='':
            with open(self.colormap_file.value, "r") as f:
                labels = f.read().splitlines()
            #Keep only values after the second occurence of "################################################"
            labels = labels[labels.index("################################################")+1:]
            labels = labels[labels.index("################################################")+1:][1:]
            choices={'choices':['0'],"key":['0 - Background']}
            colormap={0:np.array([0,0,0,0])}
            for label in labels:
                label=label.split()
                val=label[0]
                color=np.array(label[1:4]).astype('float32')/255.
                name=' '.join(label[7:]).replace('"','')
                choices['key'].append(str(val)+' - '+name)
                choices['choices'].append(str(val))
                colormap[int(val)]=np.concatenate([color,np.array([1.],dtype='float32')])
            #Get napari.layers.Labels from self.labels_layer
            # label_layer=[os.path.basename(x.name) for x in self.viewer.layers if isinstance(x,napari.layers.Labels)] #and x.name==self.labels_layer.value][0]
            # print(label_layer)
            layer=self.labels_layer.value
            for layer in [x for x in self.viewer.layers if isinstance(x,napari.layers.Labels)]:
                layer.color=colormap
            key=choices['key']
            #Convert key as a callable function that takes choices['choices'] as input and returns key
            choices['key']=lambda x: key[choices['choices'].index(x)]
            self.label_names.choices=choices
    
    def update_selected_label(self):
        print(self.label_names.value)
        layer=self.labels_layer.value
        for layer in [x for x in self.viewer.layers if isinstance(x,napari.layers.Labels)]:
            layer.selected_label=int(self.label_names.value)
            



            

#send get request to 10.29.225.156:5000/list_ckpts
#return list of ckpts

