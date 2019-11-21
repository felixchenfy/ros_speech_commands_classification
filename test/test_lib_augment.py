
import copy
import matplotlib.pyplot as plt

if True:  # Add ROOT and import my libraries.
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__)) + \
        "/../"  # Root of the project.
    sys.path.append(ROOT)

    import utils.lib_datasets as lib_datasets
    import utils.lib_augment as lib_augment

''' =========================================================== '''
# Set up data augmentation.

Aug = lib_augment.Augmenter
aug = Aug([

    # shift data for 0~0.2 percent of the total length
    Aug.Shift(rate=(0, 0.2), keep_size=False),

    Aug.PadZeros(time=(0, 0.3)),  # pad zeros at one side for 0~0.3 seconds

    Aug.Amplify(rate=(0.2, 1.5)),  # amplify loudness by 0.2~1.5

    Aug.PlaySpeed(rate=(0.7, 1.3), keep_size=False),  # change play speed

    Aug.Noise(  # Superpose noise.
        # (Noise files are pre-load and normalized)
        noise_folder="data/noises/", prob_noise=1.0, intensity=(0, 0.7)),

], prob_to_aug=1.0,  # probability to do this augmentation
)

''' =========================================================== '''
# Read audio and do two augmentations.

audio_1 = lib_datasets.AudioClass(filename="test_data/audio_front.wav")

audio_2 = copy.deepcopy(audio_1)
aug(audio_2)  # Augment audio !!!

audio_3 = copy.deepcopy(audio_1)
aug(audio_3)  # Augment audio !!!

''' =========================================================== '''
# Plot.

plt.figure(figsize=(16, 5))

plt.subplot(131)
audio_1.plot_audio(ax=plt.gca())
plt.title("Raw audio")

plt.subplot(132)
audio_2.plot_audio(ax=plt.gca())
plt.title("Augmentation 1")

plt.subplot(133)
audio_3.plot_audio(ax=plt.gca())
plt.title("Augmentation 2")

''' =========================================================== '''

# Play audios.
audio_1.play_audio()
audio_2.play_audio()
audio_3.play_audio()

# Save to files.
audio_1.write_to_file("test_augment_raw_audio.wav")
audio_2.write_to_file("test_augment_result_2.wav")
audio_3.write_to_file("test_augment_result_3.wav")

plt.show()
