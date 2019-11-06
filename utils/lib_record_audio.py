# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import time
import pynput # We need pynput.keyboard
import multiprocessing
import subprocess
import librosa
import os

if 1:  # for AudioRecorder
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import argparse, tempfile, queue, sys, datetime


def reset_audio_sample_rate(filename, dst_sample_rate):
    ''' Reset the sample rate of an audio file.
        The result will overwrite the original one.
    '''
    # dst_sample_rate = 16000, see "def stop_record"
    data, sample_rate = sf.read(filename)
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate
    sf.write(filename, data, sample_rate)
    # print(f"Reset sample rate to {dst_sample_rate} for the file: {filename}")


class TimerPrinter(object):
    # Print a message with a time gap of "T_gap"
    def __init__(self):
        self.prev_time = -999

    def print(self, s, T_gap):
        curr_time = time.time()
        if curr_time - self.prev_time < T_gap:
            return
        else:
            self.prev_time = curr_time
            print(s)

    def reset(self):
        self.prev_time = -999


class AudioRecorder(object):
    def __init__(self):
        self.init_settings()

    def init_settings(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument(
            "-l",
            "--list-devices",
            action="store_true",
            help="show list of audio devices and exit",
        )
        self.parser.add_argument(
            "-d",
            "--device",
            type=self.int_or_str,
            default="0",
            help="input device (numeric ID or substring)",
        )
        self.parser.add_argument("-r",
                                 "--samplerate",
                                 type=int,
                                 help="sampling rate")
        self.parser.add_argument("-c",
                                 "--channels",
                                 type=int,
                                 default=1,
                                 help="number of input channels")
        self.parser.add_argument(
            "filename",
            nargs="?",
            metavar="FILENAME",
            help="audio file to store recording to",
        )
        self.parser.add_argument("-t",
                                 "--subtype",
                                 type=str,
                                 help='sound file subtype (e.g. "PCM_24")')

        self.args = self.parser.parse_args()

        if self.args.list_devices:
            print(sd.query_devices())
            self.parser.exit(0)
        if self.args.samplerate is None:
            device_info = sd.query_devices(self.args.device, "input")
            # soundfile expects an int, sounddevice provides a float:
            self.args.samplerate = int(device_info["default_samplerate"])

    def start_record(self, folder="./"):

        # Some settings
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.filename = tempfile.mktemp(prefix=folder + "audio_" +
                                        self.get_time(),
                                        suffix=".wav",
                                        dir="")
        self.audio_time0 = time.time()

        # Start
        # self._thread_alive = True # This seems not working
        self.thread_record = multiprocessing.Process(target=self.record,
                                                     args=())
        self.thread_record.start()

    def stop_record(self, sample_rate=16000):
        """
        Input:
            sample_rate: desired sample rate. The original audio's sample rate is determined by
                the hardware configuration. Here, to achieve the desired sample rate, 
                this script will read the saved audio from file, resample it, 
                and then save it back to file.
        """

        # Stop thread
        self.thread_record.terminate()
        # self._thread_alive = False # This seems not working

        if 0:  # Print dashed lines
            print("\n\n" + "/" * 80)
            print("Complete writing audio to file:", self.filename)
            print("/" * 80 + "\n")

        # Check result
        time_duration = time.time() - self.audio_time0
        self.check_audio(time_duration)
        reset_audio_sample_rate(self.filename, sample_rate)

    def record(self):

        q = queue.Queue()

        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            new_val = indata.copy()
            # print(new_val)
            q.put(new_val)

        with sf.SoundFile(
                self.filename,
                mode="x",
                samplerate=self.args.samplerate,
                channels=self.args.channels,
                subtype=self.args.subtype,
        ) as file:
            with sd.InputStream(
                    samplerate=self.args.samplerate,
                    device=self.args.device,
                    channels=self.args.channels,
                    callback=callback,
            ):
                print("#" * 80)
                print("Start recording:")
                print("#" * 80)
                # while True and self._thread_alive:
                while True:
                    file.write(q.get())

    def check_audio(self, time_duration, MIN_AUDIO_LENGTH=0.1):
        # Delete file if it's too short
        print("\n")
        if time_duration < MIN_AUDIO_LENGTH:
            self.delete_file(self.filename)
            print("Audio is too short. It's been deleted.")
        else:
            print("Recorded audio is saved to: " + self.filename)
        print("-" * 80 + "\n\n")

    def delete_file(self, filename):
        subprocess.check_call("rm " + filename, shell=True)

    def int_or_str(self, text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def get_time(self):
        s = (str(datetime.datetime.now())[5:].replace(" ", "-").replace(
            ":", "-").replace(".", "-")[:-3])
        return s  # day, hour, seconds: 02-26-15-51-12-556


class KeyboardMonitor(object):
    # https://pypi.org/project/pynput/1.0.4/
    def __init__(self, default_key='R', is_print=False):
        self._recording_state = multiprocessing.Value('i', 0)
        self._is_print = is_print
        self._default_key = default_key.upper()
        self._thread = None
        self._prev_state, self._curr_state = False, False
        
    def _thread_keyboard_monitor(self):
        with pynput.keyboard.Listener(
                on_press=self._callback_on_press,
                on_release=self._callback_on_release) as listener:
            listener.join()
    
    def get_key_state(self):
        ss = (self._prev_state, self._curr_state)
        return ss

    def update_key_state(self):
        self._prev_state = self._curr_state
        self._curr_state = self._recording_state.value

    def start_listen(self, run_in_new_thread=False):
        ''' Start the keyboard listener '''
        if run_in_new_thread:
            self._thread = multiprocessing.Process(target=self._thread_keyboard_monitor,
                                                   args=())
            self._thread.start()
        else:
            self._thread_keyboard_monitor()

    def stop_listen(self):
        ''' Stop the keyboard listener '''
        if self._thread:
            self._thread.terminate()

    def is_kept_pressed(self):
        ''' Check if key is kept pressed '''
        return (self._prev_state, self._curr_state) == (True, True)

    def is_released(self):
        ''' Check if key is released '''
        return not self._curr_state

    def has_just_pressed(self):
        ''' Check if key has just been pressed '''
        return (self._prev_state, self._curr_state) == (False, True)

    def has_just_released(self):
        ''' Check if key has just been released '''
        return (self._prev_state, self._curr_state) == (True, False)

    def _key2char(self, key):
        try:
            return key.char
        except:
            return str(key)

    def _callback_on_press(self, key):
        key = self._key2char(key)
        if self._is_print:
            print("\nKey {} is pressed".format(key))
        if key.upper() == self._default_key:
            self._recording_state.value = 1

    def _callback_on_release(self, key):
        key = self._key2char(key)
        if self._is_print:
            print("\nKey {} is released".format(key))
        if key.upper() == self._default_key:
            self._recording_state.value = 0


if __name__ == "__main__":

    # Start keyboard listener
    keyboard = KeyboardMonitor(is_print=False)
    keyboard.start_listen(run_in_new_thread=True)

    # Set up audio recorder
    recorder = AudioRecorder()

    # Others
    tprinter = TimerPrinter()  # for print

    # Start loop
    while True:
        tprinter.print("Usage: keep pressing down 'R' to record audio",
                       T_gap=2)

        keyboard.update_key_state()
        if keyboard.has_just_pressed():

            # start recording
            recorder.start_record(folder="./data/data_tmp/")

            # wait until key release
            while not keyboard.has_just_released():
                keyboard.update_key_state()
                time.sleep(0.001)

            # stop recording
            recorder.stop_record()

        time.sleep(0.05)
