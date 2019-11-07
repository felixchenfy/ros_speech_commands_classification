# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function
''' 

` class AudioClass
A set of operations on audio.

` class AudioDataset
A class for loading audios and labels from folder for training the model by Torch.

` def synthesize_audio
Synthesize the audio given a string.

'''

if 1:  # Set path
    import sys, os
    ROOT = os.path.dirname(
        os.path.abspath(__file__)) + "/../"  # root of the project
    sys.path.append(ROOT)

import numpy as np
import cv2
import librosa
import matplotlib.pyplot as plt
from collections import namedtuple
import copy
from gtts import gTTS
import subprocess
import glob

import torch
from torch.utils.data import Dataset

if 1:  # my lib
    import utils.lib_proc_audio as lib_proc_audio
    import utils.lib_plot as lib_plot
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons


class AudioDataset(Dataset):
    ''' A dataset class for Pytorch to load data '''
    def __init__(
            self,
            data_folder="",
            classes_txt="",
            file_paths=[],
            file_labels=[],
            transform=None,
            is_cache_audio=False,
            is_cache_XY=True,  # cache features
    ):

        assert (data_folder and classes_txt) or (file_paths, file_labels
                                                 )  # input either one

        # Get all data's filename and label
        if not (file_paths and file_labels):
            file_paths, file_labels = AudioDataset.load_classes_and_data_filenames(
                classes_txt, data_folder)
        self._file_paths = file_paths
        self._file_labels = torch.tensor(file_labels, dtype=torch.int64)
        
        # Data augmentation methods are saved inside the `transform`
        self._transform = transform

        # Cache computed data
        self._IS_CACHE_AUDIO = is_cache_audio
        self._cached_audio = {}  # idx : audio
        self._IS_CACHE_XY = is_cache_XY
        self._cached_XY = {}  # idx : (X, Y). By default, features will be cached

    @staticmethod
    def load_classes_and_data_filenames(classes_txt, data_folder):
        '''
        Load classes names and all training data's file_paths.
        Arguments:
            classes_txt {str}: filepath of the classes.txt
            data_folder {str}: path to the data folder.
                The folder should contain subfolders named as the class name.
                Each subfolder contain many .wav files.
        '''
        # Load classes
        with open(classes_txt, 'r') as f:
            classes = [l.rstrip() for l in f.readlines()]

        # Based on classes, load all filenames from data_folder
        file_paths = []
        file_labels = []
        for i, label in enumerate(classes):
            folder = data_folder + "/" + label + "/"

            names = lib_commons.get_filenames(folder, file_types="*.wav")
            labels = [i] * len(names)

            file_paths.extend(names)
            file_labels.extend(labels)

        print("Load data from: ", data_folder)
        print("\tClasses: ", ", ".join(classes))
        return file_paths, file_labels

    def __len__(self):
        return len(self._file_paths)

    def get_audio(self, idx):
        ''' Load (idx)th audio, either from cached data, or from disk '''
        if idx in self.cached_audio:  # load from cached
            audio = copy.deepcopy(self.cached_audio[idx])  # copy from cache
        else:  # load from file
            filename = self._file_paths[idx]
            audio = AudioClass(filename=filename)
            # print(f"Load file: {filename}")
            self.cached_audio[idx] = copy.deepcopy(audio)  # cache a copy
        return audio

    def __getitem__(self, idx):

        timer = lib_commons.Timer()

        # -- Load audio
        if self._IS_CACHE_AUDIO:
            audio = self.get_audio(idx)
            print("{:<20}, len={}, file={}".format("Load audio from file",
                                                   audio.get_len_s(),
                                                   audio.filename))
        else:  # load audio from file
            if (idx in self._cached_XY) and (not self._transform):
                # if (1) audio has been processed, and (2) we don't need data augumentation,
                # then, we don't need audio data at all. Instead, we only need features from self._cached_XY
                pass
            else:
                filename = self._file_paths[idx]
                audio = AudioClass(filename=filename)

        # -- Compute features
        is_read_features_from_cache = (self._IS_CACHE_XY) and (
            idx in self._cached_XY) and (not self._transform)

        # Read features from cache:
        #   If already computed, and no augmentatation (transform), then read from cache
        if is_read_features_from_cache:
            X, Y = self._cached_XY[idx]

        # Compute features:
        #   if (1) not loaded, or (2) need new transform
        else:
            # Do transform (augmentation)
            if self._transform:
                audio = self._transform(audio)
                # self._transform(audio) # this is also good. Transform (Augment) is done in place.

            # Compute mfcc feature
            audio.compute_mfcc(n_mfcc=12)  # return mfcc

            # Compose X, Y
            X = torch.tensor(
                audio.mfcc.T,
                dtype=torch.float32)  # shape=(time_len, feature_dim)
            Y = self._file_labels[idx]

            # Cache
            if self._IS_CACHE_XY and (not self._transform):
                self._cached_XY[idx] = (X, Y)

        # print("{:>20}, len={:.3f}s, file={}".format("After transform", audio.get_len_s(), audio.filename))
        # timer.report_time(event="Load audio", prefix='\t')
        return (X, Y)


class AudioClass(object):
    ''' A wrapper around the audio data
        to provide easy access to common operations on audio data.
    '''
    def __init__(self, data=None, sample_rate=None, filename=None, n_mfcc=12):
        if filename:
            self.data, self.sample_rate = lib_io.read_audio(
                filename, dst_sample_rate=None)
        elif (len(data) and sample_rate):
            self.data, self.sample_rate = data, sample_rate
        else:
            assert 0, "Invalid input. Use keyword to input either (1) filename, or (2) data and sample_rate"

        self.mfcc = None
        self.n_mfcc = n_mfcc  # feature dimension of mfcc
        self.mfcc_image = None
        self.mfcc_histogram = None

        # Record info of original file
        self.filename = filename
        self.original_length = len(self.data)

    def get_len_s(self):  # audio length in seconds
        return len(self.data) / self.sample_rate

    def _check_and_compute_mfcc(self):
        if self.mfcc is None:
            self.compute_mfcc()

    def resample(self, new_sample_rate):
        self.data = librosa.core.resample(self.data, self.sample_rate,
                                          new_sample_rate)
        self.sample_rate = new_sample_rate

    def compute_mfcc(self, n_mfcc=None):
        # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html

        # Check input
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
        if self.n_mfcc is None:
            self.n_mfcc = n_mfcc

        # Compute
        self.mfcc = lib_proc_audio.compute_mfcc(self.data, self.sample_rate,
                                                n_mfcc)

    def compute_mfcc_histogram(
            self,
            bins=10,
            binrange=(-50, 200),
            col_divides=5,
    ):
        ''' Function:
                Divide mfcc into $col_divides columns.
                For each column, find the histogram of each feature (each row),
                    i.e. how many times their appear in each bin.
            Return:
                features: shape=(feature_dims, bins*col_divides)
        '''
        self._check_and_compute_mfcc()
        self.mfcc_histogram = lib_proc_audio.calc_histogram(
            self.mfcc, bins, binrange, col_divides)

        self.args_mfcc_histogram = (  # record parameters
            bins,
            binrange,
            col_divides,
        )

    def compute_mfcc_image(
            self,
            row=200,
            col=400,
            mfcc_min=-200,
            mfcc_max=200,
    ):
        ''' Convert mfcc to an image by converting it to [0, 255]'''
        self._check_and_compute_mfcc()
        self.mfcc_img = lib_proc_audio.mfcc_to_image(self.mfcc, row, col,
                                                     mfcc_min, mfcc_max)

    # It's difficult to set this threshold, better not use this funciton.
    def remove_silent_prefix(self, threshold=50, padding_s=0.5):
        ''' Remove the silence at the beginning of the audio data. '''

        l0 = len(self.data) / self.sample_rate

        func = lib_proc_audio.remove_silent_prefix_by_freq_domain
        self.data, self.mfcc = func(self.data,
                                    self.sample_rate,
                                    self.n_mfcc,
                                    threshold,
                                    padding_s,
                                    return_mfcc=True)

        l1 = len(self.data) / self.sample_rate
        print(f"Audio after removing silence: {l0} s --> {l1} s")

    # --------------------------- Plotting ---------------------------
    def plot_audio(self, plt_show=False, ax=None):
        lib_plot.plot_audio(self.data, self.sample_rate, ax=ax)
        if plt_show: plt.show()

    def plot_mfcc(self, method='librosa', plt_show=False, ax=None):
        self._check_and_compute_mfcc()
        lib_plot.plot_mfcc(self.mfcc, self.sample_rate, method, ax=ax)
        if plt_show: plt.show()

    def plot_audio_and_mfcc(self, plt_show=False, figsize=(12, 5)):
        plt.figure(figsize=figsize)

        plt.subplot(121)
        lib_plot.plot_audio(self.data, self.sample_rate, ax=plt.gca())

        plt.subplot(122)
        self._check_and_compute_mfcc()
        lib_plot.plot_mfcc(self.mfcc,
                           self.sample_rate,
                           method='librosa',
                           ax=plt.gca())

        if plt_show: plt.show()

    def plot_mfcc_histogram(self, plt_show=False):
        if self.mfcc_histogram is None:
            self.compute_mfcc_histogram()

        lib_plot.plot_mfcc_histogram(self.mfcc_histogram,
                                     *self.args_mfcc_histogram)
        if plt_show: plt.show()

    def plot_mfcc_image(self, plt_show=False):
        if self.mfcc_image is None:
            self.compute_mfcc_image()
        plt.show(self.mfcc_img)
        plt.title("mfcc image")
        if plt_show: plt.show()

    # --------------------------- Input / Output ---------------------------
    def write_to_file(self, filename):
        lib_io.write_audio(filename, self.data, self.sample_rate)

    def play_audio(self):
        lib_io.play_audio(data=self.data, sample_rate=self.sample_rate)


def synthesize_audio(text,
                     sample_rate=16000,
                     lang='en',
                     tmp_filename=".tmp_audio_from_SynthesizedAudio.wav",
                     is_print=False):
    '''
    Synthesize the audio of the text
    Arguments:
        text {str}: a word to be converted to audio.
        lang {str}: language
        tmp_filename {str}: path to save a temporary file for converting audio.
    Return:
        audio {AudioClass}
    '''
    # Create audio
    assert lang in ['en', 'en-uk', 'en-au',
                    'en-in']  # 4 types of acsents to choose
    if is_print:
        print(f"Synthesizing audio for '{text}'...", end=' ')
    tts = gTTS(text=text, lang=lang)

    # Save to file and load again
    tts.save(tmp_filename)
    data, sample_rate = librosa.load(
        tmp_filename)  # has to be read by librosa, not soundfile
    subprocess.call(["rm", tmp_filename])
    if is_print: print("Done!")

    # Convert to my audio class
    audio = AudioClass(data=data, sample_rate=sample_rate)
    audio.resample(sample_rate)

    return audio


def shout_out_result(audio_filepath,
                     predicted_label,
                     middle_word="is",
                     cache_folder="data/examples/"):
    '''
    Play three audios in sequence: audio_filepath, middle_word, predicted_label.
    For example:
        Three arguments are:
            audio_filepath = "dog.wav", which contains the word "dog";
            middle_word = "is";
            predicted_label = "cat";
        Then the pronounced setence is: "dog is cat"
    '''

    if not os.path.exists(cache_folder):  # create folder
        os.makedirs(cache_folder)

    fname_preword = cache_folder + middle_word + ".wav"  # create file
    if not os.path.exists(fname_preword):
        synthesize_audio(text=middle_word,
                         is_print=True).write_to_file(filename=fname_preword)

    fname_predict = cache_folder + predicted_label + ".wav"  # create file
    if not os.path.exists(fname_predict):
        synthesize_audio(text=predicted_label,
                         is_print=True).write_to_file(filename=fname_predict)

    lib_io.play_audio(filename=audio_filepath)
    lib_io.play_audio(filename=fname_preword)
    lib_io.play_audio(filename=fname_predict)


def get_wav_filenames(data_folder, suffix=".wav"):
    ''' Get all wav files under the folder;
    '''
    if suffix[0] != ".":
        suffix = "." + suffix
    if os.path.isdir(data_folder):
        filenames = glob.glob(data_folder + "/*" + suffix)
        if not filenames:
            raise RuntimeError("No .wav files in folder: " + data_folder)
    elif suffix in data_folder:
        filenames = [data_folder]
    else:
        raise ValueError('Wrong data_folder. Only .wav file is supported')
    return filenames


if __name__ == "__main__":

    def test_Class_AudioData():
        audio = AudioClass(filename="test_data/audio_numbers.wav")
        audio.plot_audio()
        audio.plot_mfcc()
        audio.plot_mfcc_histogram()

        plt.show()
        # audio.play_audio()

    def test_synthesize_audio():
        texts = ["hello"]
        texts = lib_io.read_list("config/classes_kaggle.names")
        import os, sys
        if not os.path.exists("output/"):
            os.makedirs("output/")
        for text in texts:
            print("=" * 80)
            print("Synthesizing " + text + " ...")
            audio = synthesize_audio(text, is_print=True)
            audio.play_audio()
            # audio.write_to_file(f"synthesized_audio_{text}.wav")
            audio.write_to_file(f"output/{text}.wav")

    def main():
        # test_Class_AudioData()
        test_synthesize_audio()

    main()