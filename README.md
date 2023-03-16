# napari-labelprop-remote

3D semi-automatic segmentation using deep registration-based 2D label propagation
-----------------------------------------------------------------------------------------------

![](https://raw.githubusercontent.com/nathandecaux/labelprop/master/docs/client_server.drawio.svg)

## Installation

### Server

Install [LabelProp](https://github.com/nathandecaux/labelprop)

To install it with CUDA 11.1 :

    git clone https://github.com/nathandecaux/labelprop
    cd labelprop
    pip install torch==1.10.2  --extra-index-url https://download.pytorch.org/whl/cu111
    pip install -e .

### Client
First, install [napari](
https://napari.org/tutorials/fundamentals/installation.html):
    
        pip install napari[all]


Then, to install the napari plugin :

    git clone https://github.com/nathandecaux/napari-labelprop-remote.git
    cd napari-labelprop-remote
    pip install -e .
