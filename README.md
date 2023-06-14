# napari-labelprop-remote

3D semi-automatic segmentation using deep registration-based 2D label propagation
---------------------------------------------------------------------------------

<!-- insert image and center it -->

<p align="center">
  <img src="https://github.com/nathandecaux/labelprop.github.io/raw/main/client_server.drawio.svg" width="600">
</p>

## Installation

### Server

Install [LabelProp](https://github.com/nathandecaux/labelprop)

To install it with CUDA 11.8 :

    git clone https://github.com/nathandecaux/labelprop
    cd labelprop
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -e .

### Client

First, install [napari](https://napari.org/tutorials/fundamentals/installation.html):

    pip install napari[all]

Then, to install the napari plugin :

    git clone https://github.com/nathandecaux/napari-labelprop-remote.git
    cd napari-labelprop-remote
    pip install -e .

## Usage

### Server

Start the server with the following command :

    labelprop launch-server [--addr,-a`<HOST>`] [--port,-p `<PORT>`]

This will start a Flask web server on the host `<HOST>` and port `<PORT>`. The default values are `0.0.0.0` and `5000`.

### Client

##### Setup

Start napari and open the plugin with the following command :

    napari

Then, reach the plugin in the menu bar :

    Plugins > napari-labelprop-remote > Configure

Fill the fields with the host and port of the server. Set `localhost` as host if you have set the server in the same machine. Then, click on the `Configure Server` button. Once the server is configured, you will be able to set the server-side checkpoint directory. This is the directory where the server will save the checkpoints. The default value is `path/to/labelprop/checkpoints`

##### Training

To train a model, reach the plugin in the menu bar :

    Plugins > napari-labelprop-remote > Training

Fill the fields with the following information :

- `image` : Select a loaded napari.layers.Image layer to segment
- `labels` : Select a loaded napari.layers.Labels layer with the initial labels
- `hints` : Select a loaded napari.layers.Labels layer with scribbled pseudo labels
- `pretrained checkpoint` : Select a pretrained checkpoint from the server-side checkpoint directory
- `shape` : Set the shape of slices to use for training and inference
- `z axis` : Set the axis to use for the propagation dimension
- `max epochs` : Set the maximum number of epochs to train the model
- `checkpoint name` : Set the name of the checkpoint to save on the server-side checkpoint directory
- `criteria` : Defines the criteria used to weight each direction of propagation `ncc = normalized cross correlation (slow but smooth), distance = distance to the nearest label (fast but less accurate)`
- `reduction` : When using ncc, defines the reduction to apply to the ncc map `mean / local_mean / none`. Default is `none`
- `gpu` : Set if whether to use the GPU or not. Default is `True` (GPU). GPU:0 is used by default. To use another GPU, set the `CUDA_VISIBLE_DEVICES` environment variable before launching napari.

Then, click on the `Run` button. The training will start and the progress will be displayed in the server console. Once the training is done, the checkpoint will be saved on the server-side checkpoint directory. Napari will display the forward (`propagated_up`) and backward (`propagated_down`) propagated labels and the fused labels (`propagated_fused`).

##### Inference

To run inference on a model, reach the plugin in the menu bar :

    Plugins > napari-labelprop-remote > Inference

Fill the fields like in the training section. Then, click on the `Run` button.

##### Set label colormap

To set the colormap of the labels, reach the plugin in the menu bar :

    Plugins > napari-labelprop-remote > Set label colormap

Let you load a `Label Description` file from ITKSNAP and set it to the highlighted label layer.

## Demo (inference)

<p align="center">
  <img src="https://github.com/nathandecaux/labelprop.github.io/raw/main/demo_cut.gif" width="600">
</p>
