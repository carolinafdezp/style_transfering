# DLAI Style Transfering - Team 1 

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

## Running a trained model (stylize.py)

Example of execution:


## Notes

* I tried to stay as faithful as possible to the paper and the author's implementation. This includes the decoder architecture, default hyperparams, image preprocessing, use of reflection padding in all conv layers, and bilinear upsampling + conv instead of transposed convs in the decoder. The latter two techniques help to avoid border artifacts and checkerboard patterns as described in https://distill.pub/2016/deconv-checkerboard/.



## References
[AdaIN-TF] (https://github.com/eridgd/AdaIN-TF)
