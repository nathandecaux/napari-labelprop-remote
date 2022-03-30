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
def inference(image: "napari.types.ImageData", labels: "napari.types.LabelsData",z_axis: int, label : int, checkpoint='') -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """

    # shape=torch.load(checkpoint)['hyper_parameters']['shape'][0]
    # if label==0: label='all'
    # Y_up, Y_down, Y_fused = propagate_from_ckpt(
    #     image, labels, checkpoint, z_axis=z_axis,lab=label,shape=shape)
    # return [((Y_up).astype(int), {'name': 'propagated_up'}, 'labels'), ((Y_down).astype(int), {'name': 'propagated_down'}, 'labels'), ((Y_fused).astype(int), {'name': 'propagated_fused'}, 'labels')]


#send get request to 10.29.225.156:5000/list_ckpts
#return list of ckpts

