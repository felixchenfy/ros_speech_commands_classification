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
if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import torch 
import argparse
import glob 

if 1: # my lib
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons
    import utils.lib_datasets as lib_datasets
    import utils.lib_augment as lib_augment
    import utils.lib_ml as lib_ml
    import utils.lib_rnn as lib_rnn
 
# ------------------------------------------------------------------------



def setup_classifier(load_weight_from):
    model_args = lib_rnn.set_default_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = lib_rnn.create_RNN_model(model_args, load_weight_from)
    return model

def setup_classes_labels(load_classes_from, model):
    classes = lib_io.read_list(load_classes_from)
    print(f"{len(classes)} classes: {classes}")
    model.set_classes(classes)


def main(args):
    
    model = setup_classifier(
        load_weight_from=args.weight_path)
    
    setup_classes_labels(
        load_classes_from=args.classes_path,
        model=model)
    
    filenames = lib_datasets.get_wav_filenames(
        data_folder=args.data_folder, suffix=".wav")
    
    print("\nStart predicting audio label:\n")
    
    for i, name in enumerate(filenames):
        audio = lib_datasets.AudioClass(filename=name)
        label = model.predict_audio_label(audio)
        
        print("{:03d}th file: Label = {:<10}, Filename = {}".format(
            i, label, name))
     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--weight_path', type=str, help='path to the pretrained weights',
                        default="weights/my.ckpt")
    parser.add_argument('--classes_path', type=str, help='path to classes.names',
                        default="config/classes.names")
    parser.add_argument('--data_folder', type=str, help='path to an .wav file, or to a folder containing .wav files')
    
    args = parser.parse_args()
    main(args)
        