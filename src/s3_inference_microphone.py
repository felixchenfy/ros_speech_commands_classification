# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

'''
$ python src/s3_inference_microphone.py -h
$ python src/s3_inference_microphone.py --device 0
'''       

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2
import librosa
import matplotlib.pyplot as plt 
from collections import namedtuple
import types
import time 
# import argparse # do not import argparse. It conflicts with the lib_record_audio
import torch 
import torch.nn as nn

if 1: # my lib
    import utils.lib_commons as lib_commons
    import utils.lib_rnn as lib_rnn
    import utils.lib_augment as lib_augment
    import utils.lib_datasets as lib_datasets
    import utils.lib_ml as lib_ml
    import utils.lib_io as lib_io
    from utils.lib_record_audio import * # argparse comes crom here

# -------------------------------------------------
# -- Settings
SAVE_AUDIO_TO = "data/data_tmp/"
PATH_TO_WEIGHTS = "weights/my.ckpt"
PATH_TO_CLASSES = "config/classes.names"

# -------------------------------------------------

def setup_classifier(weight_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_args = lib_rnn.set_default_args()
    model = lib_rnn.create_RNN_model(model_args, weight_file_path)
    if 0: # random test
        label_index = model.predict(np.random.random((66, 12)))
        print("Label index of a random feature: ", label_index)
        exit("Complete test.")
    return model

def setup_classes_labels(classes_txt, model):
    classes = lib_io.read_list(classes_txt)
    print(f"{len(classes)} classes: {classes}")
    model.set_classes(classes)
    
    
def inference_from_microphone():
    
    # Setup model
    model = setup_classifier(
        weight_file_path=PATH_TO_WEIGHTS)
    
    setup_classes_labels(
        classes_txt=PATH_TO_CLASSES,
        model=model)
    
    # Start keyboard listener
    keyboard = KeyboardMonitor(is_print=False)
    keyboard.start_listen(run_in_new_thread=True)

    # Set up audio recorder
    recorder = AudioRecorder()

    # Others
    tprinter = TimerPrinter() # for print

    # Start loop
    cnt_voice = 0
    while True:
        tprinter.print("Usage: keep pressing down 'R' to record audio", T_gap=20)

        keyboard.update_key_state()
        if keyboard.has_just_pressed():
            cnt_voice += 1
            print("Record {}th voice".format(cnt_voice))
            
            # start recording
            recorder.start_record(folder=SAVE_AUDIO_TO) 

            # wait until key release
            while not keyboard.has_just_released():
                keyboard.update_key_state()
                time.sleep(0.001)

            # stop recording
            recorder.stop_record()

            # Do inference
            audio = lib_datasets.AudioClass(filename=recorder.filename)
            predicted_label = model.predict_audio_label(audio)
                
            print("\nAll word labels: {}".format(model.classes))
            print("\nPredicted label: {}\n".format(predicted_label))

            # Shout out the results. e.g.: one is two
            lib_datasets.shout_out_result(recorder.filename, predicted_label,
                    middle_word="is",
                    cache_folder="data/examples/")
                
            # reset for better printing
            print("\n")
            tprinter.reset()
        
        time.sleep(0.1)
        
if __name__=="__main__":
    inference_from_microphone()
        