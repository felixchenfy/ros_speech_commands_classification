# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import time
import pynput  # We need pynput.keyboard
import multiprocessing
import subprocess
import librosa
import os
import warnings
import threading

if 1:  # for AudioRecorder
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import argparse
    import tempfile
    import queue
    import sys
    import datetime


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
    ''' A class for printing message with time control:
        If previous `print` is within T_gap seconds, 
        then the currrent `print` prints nothing. 
    '''

    def __init__(self, print_period):
        self._prev_time = -999
        self._t_print_gap = print_period

    def print(self, s):
        curr_time = time.time()
        if curr_time - self._prev_time < self._t_print_gap:
            return
        else:
            self._prev_time = curr_time
            print(s)

    def reset(self):
        self._prev_time = -999.0


class AudioRecorder(object):
    '''
    A class for recording audio from the laptop's microphone.
    The recorded audio will be saved to disk.

    I copied this from somewhere, but I forgot where it is.

    For the unit test, please see: test_KeyboardInputFromTerminal_and_AudioRecorder()

    '''

    def __init__(self):
        self.init_settings()
        self._set_filename = lambda folder: tempfile.mktemp(
            prefix=folder + "audio_" + self.get_time(), suffix=".wav", dir="")

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

        self.filename = self._set_filename(folder)
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


class KeyboardInputFromTerminal(object):
    '''
    Detect keyboard press/release events.
    For more details, please see: https://pypi.org/project/pynput/1.0.4/
    For the unit test, please see: test_KeyboardInputFromTerminal_and_AudioRecorder()
    '''

    def __init__(self, hotkey='R', run_in_new_thread=True, is_print=False):

        self._IS_PRINT = is_print
        self._HOTKEY = hotkey.upper()

        self._recording_state = multiprocessing.Value('i', 0)
        self._thread = None
        self._prev_state, self._curr_state = False, False
        self._RUN_IN_NEW_THREAD = run_in_new_thread

        # When the class instance is destroyed, this flag will be set to False.
        self._is_alive = True

        # Keypress state
        self._prev_state, self._curr_state == False, False
        self._t_last_pressing = -999.0  # Last time when key is released to press
        self._t_last_releasing = -999.0  # Last time when key is pressed to released

        # Start listening
        self._start_listen(run_in_new_thread)

    def is_key_pressed(self):
        ''' Check if key is pressed '''
        return self._curr_state

    def is_key_released(self):
        ''' Check if key is released '''
        return not self._curr_state

    def is_kept_pressed(self):
        ''' Check if key is kept pressed '''
        return (self._prev_state, self._curr_state) == (True, True)

    def has_just_pressed(self, t_tolerance=0.1):
        ''' Check if key has just been pressed. '''
        return (time.time() - self._t_last_pressing) <= t_tolerance

    def has_just_released(self, t_tolerance=0.1):
        ''' Check if key has just been released '''
        return (time.time() - self._t_last_releasing) <= t_tolerance

    def get_key_state(self):
        ss = (self._prev_state, self._curr_state)
        return ss

    def __del__(self):
        self._is_alive = False

    def _thread_keyboard_monitor(self):
        with pynput.keyboard.Listener(
                on_press=self._callback_on_press,
                on_release=self._callback_on_release) as listener:
            while self._is_alive:
                time.sleep(0.1)
            listener.join()

    def _start_listen(self, run_in_new_thread):
        ''' Start the keyboard listener '''
        if run_in_new_thread:
            self._thread = threading.Thread(
                target=self._thread_keyboard_monitor, args=())
            self._thread.daemon = True  # enable ctrl+c to stop program
            self._thread.start()
        else:
            self._thread_keyboard_monitor()

    # def stop_listen(self):
    #     ''' Stop the keyboard listener '''
    #     if self._thread:
    #         self._thread.terminate()

    def _key2char(self, key):
        try:
            key = key.char
        except:
            key = str(key)
        key = key.upper()
        return key

    def _callback_on_press(self, key):
        ''' Callback function when any key is pressed down. '''
        key = self._key2char(key)
        if self._IS_PRINT:
            print("\nKey {} is pressed".format(key))
        if key == self._HOTKEY:
            self._recording_state.value = 1
            self._update_key_state()

    def _callback_on_release(self, key):
        ''' Callback function when any key is released. '''
        key = self._key2char(key)
        if self._IS_PRINT:
            print("\nKey {} is released".format(key))
        if key == self._HOTKEY:
            self._recording_state.value = 0
            self._update_key_state()

    def _update_key_state(self):
        self._prev_state = self._curr_state
        self._curr_state = self._recording_state.value == 1
        if (self._prev_state, self._curr_state) == (False, True):
            self._t_last_pressing = time.time()
        if (self._prev_state, self._curr_state) == (True, False):
            self._t_last_releasing = time.time()


def test_KeyboardInputFromTerminal_and_AudioRecorder():
    ''' Use keypress to start/stop recording audio.
    The audio will be saved to `dst_folder`.
    '''
    dst_folder = "./data/data_tmp/"

    # Start keyboard listener
    keyboard = KeyboardInputFromTerminal(
        hotkey="R", is_print=False, run_in_new_thread=True)

    # Set up audio recorder
    recorder = AudioRecorder()

    # Others
    timer_printer = TimerPrinter(print_period=2.0)  # for print

    # Start loop
    while True:
        timer_printer.print("Usage: keep pressing down 'R' to record audio")
        if keyboard.is_key_pressed():
            recorder.start_record(folder=dst_folder)  # Start record
            while not keyboard.is_key_released():  # Wait for key released
                time.sleep(0.001)
            recorder.stop_record()  # Stop record
        time.sleep(0.01)


if __name__ == "__main__":
    test_KeyboardInputFromTerminal_and_AudioRecorder()
