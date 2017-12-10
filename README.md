# AdaIN-TF

<p align='center'>
	<img src='examples/gilbert.gif'>
</p>

This is a TensorFlow/Keras implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868).


## Requirements

* Python 3.x
* tensorflow 1.2.1+
* keras 2.0.x
* torchfile 

Optionally:

* OpenCV with contrib modules (for `webcam.py`)
  * MacOS install http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/
  * Linux install http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
* ffmpeg (for video stylization)

## Training

1. Download [MS COCO images](http://mscoco.org/dataset/#download) and [Wikiart images](https://www.kaggle.com/c/painter-by-numbers).

2. Download VGG19 model: `bash models/download_models.sh`

3. `python train.py --content-path /path/to/coco --style-path /path/to/wikiart --batch-size 8 --content-weight 1 --style-weight 1e-2 --tv-weight 0 --checkpoint /path/to/checkpointdir --learning-rate 1e-4 --lr-decay 1e-5`

3. Monitor training with TensorBoard: `tensorboard --logdir /path/to/checkpointdir`

## Running a trained model

To stylize images captured from a webcam:

`python webcam.py --checkpoint /path/to/checkpointdir --style-path /path/to/wikiart`

This script contains a few options:

* `--source`  Use a different camera input
* `--width` and `--height`  Set the size of camera frames
* `--video-out`  Write stylized frames to .mp4 out path
* `--fps`  Frames Per Second for video out
* `--scale`  Resize content images by this factor before stylizing
* `--keep-colors`  Apply CORAL transform to preserve colors of content
* `--device`  Device to perform compute on, default `/gpu:0`
* `--style-size`  Resize small side of style image to this before cropping a 256x256 square
* `--alpha`  Control blending of content features + AdaIN transformed features
* `--concat`  Append the style image to the stylized output
* `--interpolate`  Interpolate between AdaIN features of two random images
* `--noise`  Generate textures from random noise image instead of webcam
* `--random`  Load a new random image every # of frames

There are also three keyboard shortcuts:

* `r`  Load random image from style folder
* `c`  Toggle color preservation
* `q`  Quit cleanly and close streams

Additionally, `stylize.py` will stylize image files. The options are the same as for the webcam script with the addition of `--content-path`, which can be a single image file or folder. Each style in `--style-path` will be applied to each content image. 

## Notes

* I tried to stay as faithful as possible to the paper and the author's implementation. This includes the decoder architecture, default hyperparams, image preprocessing, use of reflection padding in all conv layers, and bilinear upsampling + conv instead of transposed convs in the decoder. The latter two techniques help to avoid border artifacts and checkerboard patterns as described in https://distill.pub/2016/deconv-checkerboard/.
* The same normalised VGG19 is also used with weights loaded from `vgg_normalised.t7` and then translated into Keras layers.  A version that uses a modified `keras.applications.VGG19` can be found in the `vgg_keras` branch.
* `coral.py` implements [CORellation ALignment](https://arxiv.org/abs/1612.01939) to transfer colors from the content image to the style image in order to preserve colors in the stylized output. The default method uses numpy, and I have also translated the author's CORAL code from Torch to PyTorch.

## Acknowledgments

Many thanks to the author Xun Huang for the excellent [original Torch implementation](https://github.com/xunhuang1995/AdaIN-style) that saved me countless hours of frustation. I also drew inspiration from Jon Rei's [TF implementation for the .t7 pre-trained decoder](https://github.com/jonrei/tf-AdaIN) and borrowed the clean TF code for the AdaIN transform.


## TODO

- [x] CORAL for preserving colors
- [x] Image stylization
- [x] Docs
- [ ] Fix interpolation for webcam
- [ ] Pre-trained model
- [ ] Pre-compute style encoding means/stds
- [ ] Video processing
- [ ] Webcam style window threading
- [x] Keras VGG
