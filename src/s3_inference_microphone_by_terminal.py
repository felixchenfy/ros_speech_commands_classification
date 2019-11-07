# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

'''
$ python src/s3_inference_microphone_by_terminal.py -h
$ python src/s3_inference_microphone_by_terminal.py --device 0
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
SRC_WEIGHT_PATH = ROOT + "weights/my.ckpt"
SRC_CLASSES_PATH = ROOT + "config/classes.names"
DST_AUDIO_FOLDER = ROOT + "data/data_tmp/"

# -------------------------------------------------
# -- Main function    
def inference_from_microphone():
    
    # Setup model
    model, classes = lib_rnn.setup_default_RNN_model(
        weight_filepath=SRC_WEIGHT_PATH, classes_txt=SRC_CLASSES_PATH)
    print(f"{len(classes)} classes: {classes}")
    
    # Start keyboard listener
    keyboard = KeyboardInputFromTerminal(
        hotkey="R", is_print=False, run_in_new_thread=True)

    # Set up audio recorder
    recorder = AudioRecorder()

    # Others
    timer_printer = TimerPrinter(print_period=2.0)  # for print

    # Start loop
    cnt_voice = 0
    while True:
        timer_printer.print("Usage: keep pressing down 'R' to record audio")
        if keyboard.is_key_pressed():
            cnt_voice += 1
            print("\nRecord {}th voice".format(cnt_voice))
            
            # Record audio
            recorder.start_record(folder=DST_AUDIO_FOLDER)  # Start record
            while not keyboard.is_key_released():  # Wait for key released
                time.sleep(0.001)
            recorder.stop_record()  # Stop record

            # Do inference
            audio = lib_datasets.AudioClass(filename=recorder.filename)
            predicted_label = model.predict_audio_label(audio)
            print("\nAll word labels: {}".format(model.classes))
            print("\nPredicted label: {}\n".format(predicted_label))

            # Shout out the results. e.g.: one is two
            lib_datasets.shout_out_result(recorder.filename, predicted_label,
                    middle_word="is",
                    cache_folder="data/examples/")
                
        time.sleep(0.1)
        
if __name__=="__main__":
    inference_from_microphone()
        