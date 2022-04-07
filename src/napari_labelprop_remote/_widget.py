"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory,magicgui
from magicgui.widgets import Select
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

def get_ckpts():
    url = 'http://10.29.225.156:5000/list_ckpts'
    response = requests.get(url).text
    return response.split(',')


def inject_items(d, items):
    for i,v in enumerate(items):
        d[str(v)] = v

class Checkpoints(Enum):
    inject_items(locals(), ['a','b','c'])

@magic_factory(checkpoint={'choices':['']+get_ckpts()})
def inference(image: "napari.layers.Image", labels: "napari.layers.Labels",z_axis: int, label : int, checkpoint='') -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    r=urljoin(url,'inference')
    params={'z_axis':z_axis,'label':label,'checkpoint':checkpoint}
    params=json.dumps(params).encode('utf-8')
    buf=create_buf_npz({'img':image.data,'mask':labels.data})
    response=requests.post(r,files={'arrays':buf,'params':params})
    token=response.text
    buf.close()
    while not session_exists(token):
        time.sleep(5)
    r=urljoin(url,'download_inference')
    r=r+'?token='+token
    npz_file=np.load(get_file(r),encoding = 'latin1')
    Y_up,Y_down,Y_fused=npz_file['Y_up'],npz_file['Y_down'],npz_file['Y_fused']
    return [((Y_up).astype(int), {'name': 'propagated_up','metadata':labels.metadata}, 'labels'), ((Y_down).astype(int), {'name': 'propagated_down','metadata':labels.metadata}, 'labels'), ((Y_fused).astype(int), {'name': 'propagated_fused','metadata':labels.metadata}, 'labels')]
    # shape=torch.load(checkpoint)['hyper_parameters']['shape'][0]
    # if label==0: label='all'
    # Y_up, Y_down, Y_fused = propagate_from_ckpt(
    #     image, labels, checkpoint, z_axis=z_axis,lab=label,shape=shape)
    # return [((Y_up).astype(int), {'name': 'propagated_up'}, 'labels'), ((Y_down).astype(int), {'name': 'propagated_down'}, 'labels'), ((Y_fused).astype(int), {'name': 'propagated_fused'}, 'labels')]

@magic_factory(pretrained_checkpoint={'choices':['']+get_ckpts()})
def training(image: "napari.types.ImageData", labels: "napari.types.LabelsData", pretrained_checkpoint: "napari.types.Path" = '', shape: int=256, z_axis: int=0, max_epochs: int=10,checkpoint_name='',pretraining=False) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    r=urljoin(url,'training')
    params={'pretrained_ckpt':pretrained_checkpoint,'shape':shape,'z_axis':z_axis,'max_epochs':max_epochs,'name':checkpoint_name,'pretraining':pretraining}
    params=json.dumps(params).encode('utf-8')
    buf=create_buf_npz({'img':image,'mask':labels})
    response=requests.post(r,files={'arrays':buf,'params':params})
    token=response.text
    buf.close()
    while not session_exists(token):
        time.sleep(5)
    r=urljoin(url,'download_inference')
    r=r+'?token='+token
    npz_file=np.load(get_file(r),encoding = 'latin1')
    Y_up,Y_down,Y_fused=npz_file['Y_up'],npz_file['Y_down'],npz_file['Y_fused']
    return [((Y_up).astype(int), {'name': 'propagated_up'}, 'labels'), ((Y_down).astype(int), {'name': 'propagated_down'}, 'labels'), ((Y_fused).astype(int), {'name': 'propagated_fused'}, 'labels')]

#send get request to 10.29.225.156:5000/list_ckpts
#return list of ckpts

