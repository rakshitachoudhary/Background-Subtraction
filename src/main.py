""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np
import baseline_final
import jitter_final
import illumination_final
import moving_bg_final
import ptz_final

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    
    return args


def baseline_bgs(args):
    baseline_final.baseline(args.inp_path, args.out_path, args.eval_frames)


def illumination_bgs(args):
    illumination_final.illumination(args.inp_path, args.out_path, args.eval_frames)


def jitter_bgs(args):
    jitter_final.jitter(args.inp_path, args.out_path, args.eval_frames)


def dynamic_bgs(args):
    moving_bg_final.moving_bg(args.inp_path, args.out_path, args.eval_frames)


def ptz_bgs(args):
    ptz_final.ptz(args.inp_path, args.out_path, args.eval_frames)


def main(args):
    if args.category not in "bijmp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
