# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

'''
Resample all audios under a folder to a sampling rate of 16000.
'''

import glob 
import os 
import soundfile as sf
import librosa

def get_filenames(folder, file_type):
    return glob.glob(folder + "/" + file_type)

def reset_audio_file_sample_rate(filename, dst_sample_rate=16000):
    data, sample_rate = sf.read(filename) 
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate
    sf.write(filename, data, sample_rate)
    print(f"Reset sample rate to {dst_sample_rate} for the file: {filename}")
    
folder = "./data/data_tmp/"
fnames1 = get_filenames(folder, file_type="*/*.wav")
fnames2 = get_filenames(folder, file_type="*.wav")
fnames = fnames1 + fnames2 

for name in fnames:
    reset_audio_file_sample_rate(name)
