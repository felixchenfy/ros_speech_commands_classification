# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

'''
$ python src/s4_inference_microphone_by_GUI.py -h
$ python src/s4_inference_microphone_by_GUI.py --device 0
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
    from utils.lib_gui import GuiForAudioClassification
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
    if 0: # Test with random data
        label_index = model.predict(np.random.random((66, 12)))
        print("Label index of a random feature: ", label_index)
        exit("Complete test.")
    return model

def setup_classes_labels(classes_txt, model):
    classes = lib_io.read_list(classes_txt)
    print(f"{len(classes)} classes: {classes}")
    model.set_classes(classes)
    
    
def inference_from_microphone():
    
    # Setup LSTM model
    model = setup_classifier(
        weight_file_path=PATH_TO_WEIGHTS)
    
    setup_classes_labels(
        classes_txt=PATH_TO_CLASSES,
        model=model)
    
    # GUI
    classes = lib_io.read_list(PATH_TO_CLASSES)
    gui = GuiForAudioClassification(classes, hotkey="R")

    # Set up audio recorder
    recorder = AudioRecorder()

    # Others
    timer_printer = TimerPrinter(print_period=2.0)  # for print

    # Start loop
    cnt_voice = 0
    while not gui.is_key_quit_pressed():
        timer_printer.print("Usage: keep pressing down 'R' to record audio")
        if gui.is_key_pressed():
            cnt_voice += 1
            print("\nRecord {}th voice".format(cnt_voice))
            
            # -- Record audio
            gui.enable_img1_self_updating()
            recorder.start_record(folder=SAVE_AUDIO_TO)  # Start record
            while not gui.is_key_released():  # Wait for key released
                time.sleep(0.001)
            recorder.stop_record()  # Stop record

            # -- Do inference
            audio = lib_datasets.AudioClass(filename=recorder.filename)
            probs = model.predict_audio_label_probabilities(audio)
            predicted_idx = np.argmax(probs)
            predicted_label = classes[predicted_idx]
            max_prob = probs[predicted_idx]
            print("\nAll word labels: {}".format(classes))
            print("\nPredicted label: {}, probability: {}\n".format(
                predicted_label, max_prob))
            PROB_THRESHOLD = 0.8
            final_label  = predicted_label if max_prob > PROB_THRESHOLD else "None"
            
            # -- Update the image

            # Update image1: first stop self updating, 
            # then set recording_length and voice_intensity to zero
            gui.reset_img1() 

            # Update image 2: the prediction results
            gui.set_img2(
                final_label=final_label,
                predicted_label=predicted_label, 
                probability=max_prob, 
                length=audio.get_len_s(),
                valid_length=audio.get_len_s(), # TODO: remove the silent voice,
            )
            
            # Update image 3: the probability of each class
            gui.set_img3(probabilities=probs)
            
            # -- Shout out the results. e.g.: two is one
            lib_datasets.shout_out_result(recorder.filename, final_label,
                    middle_word="is",
                    cache_folder="data/examples/")
                
        time.sleep(0.1)

    print("\n=====================================================")
    print("`def inference_from_microphone` stops")
    print("=====================================================\n")

if __name__=="__main__":
    inference_from_microphone()
        