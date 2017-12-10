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


## Training

Example of training:

`python train.py --content-path /path/to/coco --style-path /path/to/wikiart --batch-size 8 --content-weight 1 --style-weight 1e-2 --tv-weight 0 --checkpoint /path/to/checkpointdir --learning-rate 1e-4 --lr-decay 1e-5`

## Running a trained model (stylize.py)

Example of execution:


## Notes

* I tried to stay as faithful as possible to the paper and the author's implementation. This includes the decoder architecture, default hyperparams, image preprocessing, use of reflection padding in all conv layers, and bilinear upsampling + conv instead of transposed convs in the decoder. The latter two techniques help to avoid border artifacts and checkerboard patterns as described in https://distill.pub/2016/deconv-checkerboard/.



## References
[AdaIN-TF](https://github.com/eridgd/AdaIN-TF)
