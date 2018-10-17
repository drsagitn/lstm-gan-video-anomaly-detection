#!/usr/bin/env python
import argparse
import cv2
import os
import glob
import sys
import numpy as np
import scipy.io as sio
import time

def cvReadGrayImg(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

def saveOptFlowToImage(flow, basename, merge):
    if merge:
        # save x, y flows to r and g channels, since opencv reverses the colors
        cv2.imwrite(basename+'.png', flow[:,:,::-1])
    else:
        cv2.imwrite(basename+'_x.JPEG', flow[...,0])
        cv2.imwrite(basename+'_y.JPEG', flow[...,1])

def gen_multifolders(parent_folder, out_folder):
    for dir in os.listdir(parent_folder):
        dir_path = os.path.join(parent_folder, dir)
        if os.path.isdir(dir_path):
            save_path = os.path.join(out_folder, dir)
            gen_optical_flow(dir_path, save_path, True, False)




def gen_optical_flow(vid_dir, save_dir, merge, visual_debug, bound=15):
    norm_width = 500.
    bound = bound

    images = glob.glob(os.path.join(vid_dir, '*'))
    print("Processing {}: {} files... ".format(vid_dir, len(images))),
    sys.stdout.flush()
    tic = time.time()
    img2 = cvReadGrayImg(images[0])
    for ind, img_path in enumerate(images[:-1]):
        img1 = img2
        img2 = cvReadGrayImg(images[ind + 1])
        h, w = img1.shape
        fxy = int(norm_width / w)
        # normalize image size
        flow = cv2.calcOpticalFlowFarneback(
            cv2.resize(img1, None, fx=fxy, fy=fxy),
            cv2.resize(img2, None, fx=fxy, fy=fxy),
            None,
            0.5, 3, 15, 3, 7, 1.5, 0)
        # map optical flow back
        flow = flow / fxy
        # normalization
        flow = np.round((flow + bound) / (2. * bound) * 255.)
        flow[flow < 0] = 0
        flow[flow > 255] = 255
        flow = cv2.resize(flow, (w, h))

        # Fill third channel with zeros
        flow = np.concatenate((flow, np.zeros((h, w, 1))), axis=2)

        # save
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        saveOptFlowToImage(flow, os.path.join(save_dir, basename), merge)

        if visual_debug:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(cv2.imread(img_path))
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            cv2.imshow('optical flow', bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    # duplicate last frame
    basename = os.path.splitext(os.path.basename(images[-1]))[0]
    saveOptFlowToImage(flow, os.path.join(save_dir, basename), merge)
    toc = time.time()
    print("{:.2f} min, {:.2f} fps".format((toc - tic) / 60., 1. * len(images) / (toc - tic)))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--vid_dir', required=False, default='data/UCSDped1/Test/Test002')
    # parser.add_argument('--save_dir', required=False, default='data/optical_flow/Test002')
    # parser.add_argument('--bound', type=float, required=False, default=15,
    #                     help='Optical flow bounding. [-bound, bound] will be mapped to [0, 255].')
    # parser.add_argument('--merge', dest='merge', action='store_true', default=True,
    #                     help='Merge optical flow in x and y axes into RGB images rather than saving each to a grayscale image.')
    # parser.add_argument('--debug', dest='visual_debug', action='store_true',
    #                     help='Visual debugging.')
    # parser.set_defaults(visual_debug=False)
    # args = parser.parse_args()
    # gen_optical_flow(args.vid_dir, args.save_dir, args.merge, args.visual_debug)
    gen_multifolders("data/UCSDped1/Test", "data/optical_flow/UCSDped1/Test")



