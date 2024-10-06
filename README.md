# I2MVs: MultiView Video Generation from a Single Image
In this repository, we provide:

- [x] A small dataset containing [real-scanned 3D assets](https://omniobject3d.github.io/) across six categories of commonly seen clothing accessories.
- [x] A non-official [SV3D](https://stability.ai/news/introducing-stable-video-3d) training code.
- [ ] A new 3D inpainting method


## The DataSet
We collected 154 real-scanned 3D models from the [OmniObject3D](https://omniobject3d.github.io/) , manually masked varying percentages of the visible mesh faces, and re-rendered orbital videos with [Blender](https://www.blender.org/): 

![hat_003_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/hat_003_cam_01.gif)
![backpack_019_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/backpack_019_cam_01.gif)
![bumbag_004_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/bumbag_004_cam_01.gif)
![handbag_053_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/handbag_053_cam_01.gif)
![shoe_003_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/shoe_003_cam_01.gif)
![suitcase_006_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/suitcase_006_cam_01.gif)

The whole dataset with official split can be found at [Google Drive]().

The dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## SV3D Re-Implementation

### Training

Please update the [config.py](https://github.com/Jun-Pu/I2MVs/blob/main/code/configs.py) before running [train.py](https://github.com/Jun-Pu/I2MVs/blob/main/code/train.py).

### Inference

Train your own SV3D model or simply download our [checkpoints](), then run [test.py](https://github.com/Jun-Pu/I2MVs/blob/main/code/test.py) to achieve results like these:

<img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/hat_007_0026.gif" alt="GIF 1" width="200" style="display:inline;"/> <img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/hat_019_0026.gif" alt="GIF 2" width="200" style="display:inline;"/> 
<img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/hat_022_0026.gif" alt="GIF 3" width="200" style="display:inline;"/> <img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/hat_029_0026.gif" alt="GIF 4" width="200" style="display:inline;"/>

<img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/backpack_024_0026.gif" alt="GIF 5" width="200" style="display:inline;"/> <img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/handbag_007_0026.gif" alt="GIF 6" width="200" style="display:inline;"/>
<img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/handbag_012_0026.gif" alt="GIF 7" width="200" style="display:inline;"/>
<img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/handbag_039_0026.gif" alt="GIF 8" width="200" style="display:inline;"/>

<img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/shoe_003_0001.gif" alt="GIF 9" width="200" style="display:inline;"/> <img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/shoe_006_0001.gif" alt="GIF 10" width="200" style="display:inline;"/>
<img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/shoe_012_0001.gif" alt="GIF 11" width="200" style="display:inline;"/>
<img src="https://github.com/Jun-Pu/I2MVs/blob/main/demos/shoe_028_0001.gif" alt="GIF 12" width="200" style="display:inline;"/>

## Citation

     @software{I2MVs,
      author = {Yi Zhang},
      title = {I2MVs: MultiView Video Generation from a Single Image},
      month = {October},
      year = {2024},
      url = {https://github.com/Jun-Pu/I2MVs}
    }


