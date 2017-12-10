from __future__ import division, print_function

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from utils import preserve_colors_np
from utils import get_files, get_img, get_img_crop, save_img, resize_to
import scipy
from scipy.ndimage.filters import gaussian_filter
import time
from inference import AdaINference


parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('--checkpoint', type=str, help='Checkpoint directory', required=True)
parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')
parser.add_argument('--vgg-path', type=str,
                    dest='vgg_path', help='Path to vgg_normalised.t7', 
                    default='models/vgg_normalised.t7')
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
parser.add_argument('--device', type=str,
                        dest='device', help='Device to perform compute on',
                        default='/gpu:0')
parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512", default=512)
parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=256)
parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=0)
parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)
parser.add_argument('-r','--random', type=int, help="Choose # of random subset of images from style folder", default=0)
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)
args = parser.parse_args()


def main():
    start = time.time()

    # Load the AdaIN model
    ada_in = AdaINference(args.checkpoint, args.vgg_path, device=args.device)

    # Get content & style full paths
    if os.path.isdir(args.content_path):
        content_files = get_files(args.content_path)
    else: # Single image file
        content_files = [args.content_path]
    if os.path.isdir(args.style_path):
        style_files = get_files(args.style_path)
        if args.random > 0:
            style_files = np.random.choice(style_files, args.random)
    else: # Single image file
        style_files = [args.style_path]

    os.makedirs(args.out_path, exist_ok=True)

    count = 0

    ### Apply each style to each content image
    for content_fullpath in content_files:
        content_prefix, content_ext = os.path.splitext(content_fullpath)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext

        content_img = get_img(content_fullpath)
        if args.content_size > 0:
            content_img = resize_to(content_img, args.content_size)

        for style_fullpath in style_files: 
            style_prefix, _ = os.path.splitext(style_fullpath)
            style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

            style_img = get_img_crop(style_fullpath, resize=args.style_size, crop=args.crop_size)
            # style_img = get_img(style_fullpath)

            if args.keep_colors:
                style_img = preserve_colors_np(style_img, content_img)

            # if args.noise:  # Generate textures from noise instead of images
            #     frame_resize = np.random.randint(0, 256, frame_resize.shape, np.uint8)
            #     frame_resize = gaussian_filter(frame_resize, sigma=0.5)

            # Run the frame through the style network
            stylized_rgb = ada_in.predict(content_img, style_img, args.alpha)

            if args.passes > 1:
                for _ in range(args.passes-1):
                    stylized_rgb = ada_in.predict(stylized_rgb, style_img, args.alpha)

            # Stitch the style + stylized output together, but only if there's one style image
            if args.concat:
                # Resize style img to same height as frame
                style_img_resized = scipy.misc.imresize(style_img, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                stylized_rgb = np.hstack([style_img_resized, stylized_rgb])

            # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
            out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))
            # out_f = f'{content_prefix}_{style_prefix}.{content_ext}'
            
            save_img(out_f, stylized_rgb)

            count += 1
            print("{}: Wrote stylized output image to {}".format(count, out_f))

    print("Finished stylizing {} outputs in {}s".format(count, time.time() - start))


if __name__ == '__main__':
    main()
