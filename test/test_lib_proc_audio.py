'''
Test the function `lib_proc_audio.compute_mfcc`,
    which is called by `audio.compute_mfcc()`.
'''

import matplotlib.pyplot as plt

if True:  # Add ROOT and import my libraries.
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__)) + \
        "/../"  # Root of the project.
    sys.path.append(ROOT)

    import utils.lib_datasets as lib_datasets

audio = lib_datasets.AudioClass(filename="test_data/audio_front.wav")
audio.compute_mfcc()
audio.plot_audio_and_mfcc()
plt.show()
