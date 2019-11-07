# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

''' Inference one audio file, of all audio files in a folder.
 
Testing commands for this script:

    One file:
    $ python src/s5_inference_audio_file.py --data_folder test_data/audio_front.wav
    $ python src/s5_inference_audio_file.py --data_folder test_data/audio_three.wav

    A folder:
    $ python src/s5_inference_audio_file.py --data_folder data/data_train/three
    
'''

if 1:  # Set path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
    sys.path.append(ROOT)

import glob
import torch
import numpy as np
import argparse

import utils.lib_rnn as lib_rnn
import utils.lib_ml as lib_ml
import utils.lib_augment as lib_augment
import utils.lib_datasets as lib_datasets
import utils.lib_commons as lib_commons
import utils.lib_io as lib_io





# ------------------------------------------------------------------------


def main(args):

    # -- Init model
    model, classes = lib_rnn.setup_default_RNN_model(
        args.weight_path, args.classes_path)

    # -- If `data_folder` is a filename, return [data_folder].
    #    If `data_folder` is a folder, return all .wav filenames in this folder.
    filenames = lib_datasets.get_wav_filenames(
        data_folder=args.data_folder, suffix=".wav")

    # -- Classification
    print("\nStart predicting audio label:\n")

    for i, name in enumerate(filenames):
        audio = lib_datasets.AudioClass(filename=name)
        label = model.predict_audio_label(audio)

        print("{:03d}th file: Label = {:<10}, Filename = {}".format(
            i, label, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--weight_path', type=str, help='path to the pretrained weights',
                        default="weights/my.ckpt")
    parser.add_argument('--classes_path', type=str, help='path to classes.names',
                        default="config/classes.names")
    parser.add_argument('--data_folder', type=str,
                        help='path to an .wav file, or to a folder containing .wav files')

    args = parser.parse_args()
    main(args)
        