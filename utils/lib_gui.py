# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function
'''
A GUI for displaying the audio classification result.
    See `class GuiForAudioClassification`.
A unit test is given in `def test_GUI`.
'''

import cv2
import threading
import numpy as np
import time
import datetime
import json

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)


# -- Functions


def read_list(filename):
    with open(filename) as f:
        with open(filename, 'r') as f:
            data = [l.rstrip() for l in f.readlines()]
    return data


def add_black_border(img, border_width):
    r, c = img.shape[0], img.shape[1]
    img[0:border_width, :] = 0
    img[r-border_width:r, :] = 0
    img[:, 0:border_width] = 0
    img[:, c-border_width:] = 0


def get_time_string():
    ''' Get a formatted string time: `month-day-hour-minute-seconds-miliseconds`,
        such as: `02-26-15-51-12-106`.
    '''
    s = str(datetime.datetime.now())[5:].replace(
        ' ', '-').replace(":", '-').replace('.', '-')[:-3]
    return s


def get_hour_and_minute_and_seconds():
    s = get_time_string()
    s = s[6:-4].replace("-", ":")
    return s  # 15:51:12, which means "3pm 51min 12s"


class TextBoxDrawer(object):
    def __init__(self, img, x0, y0, fontsize=0.8, texts=[]):
        self._img = img
        self._x0, self._y0 = x0, y0
        self._fontsize = fontsize
        for text in texts:
            self.add_text(text)

    def add_text(self, text, font=cv2.FONT_HERSHEY_DUPLEX, thickness=1):
        ''' Draw text into the TextBox, and then change to a new line. '''

        # Change to new line
        (text_width, text_height) = cv2.getTextSize(text,
                                                    font,
                                                    fontScale=self._fontsize,
                                                    thickness=thickness)[0]
        self._y0 += text_height + int(15 * self._fontsize)

        # Draw text
        cv2.putText(self._img,
                    text, (self._x0, self._y0),
                    fontFace=font,
                    fontScale=self._fontsize,
                    color=(0, 0, 0),
                    thickness=thickness)


class GuiForAudioClassification(object):
    '''
    This class initializes a new thread which keeps displaying image by cv2.imshow()
        and keeps detecting keyboard events.
    Several interface functions are provided for 
        changing the contents of the displayed image.

    The image is composed of three sub images:
        ---------------
    h1  |  img1 |      |
        |-------| img3 |
    h2  |  img2 |      |
        ---------------
           w1      w2
    See set_img1(), set_img2(), set_img3() for more details

    Public member functions:
        set_img1()
        set_img2()
        set_img3()
        is_key_pressed()
        is_key_released()

    Keys:
        self.HOTKEY: its state is returend by self.is_key_pressed()
        q, Q: If it's pressed, self.is_q_pressed() will be True.
    '''

    def __init__(self, classes, hotkey='R', key_effective_duration=0.5,
                 h1=150, h2=340, w1=550, w2=265):
        '''
        Argument:
            hotkey {char}: self.is_key_pressed() detects the state of `hotkey`.
            key_effective_duration {float}: After a key is pressed down, 
                in the following `key_effective_duration` seconds,
                they key will be considered as pressed no matter it's released or not.
                (This is set due to the unfortunate fact that
                 cv2.waitKey might return 0 even when the key is kept pressed down.)
                A suggested value tested on my Ubuntu 18.04 is 0.5.
        '''
        # Variables
        self._CLASSES = classes
        self._HOTKEY = hotkey
        self._KEY_EFFECTIVE_DURATION = key_effective_duration
        self._result_key = ''
        self._t_prev_pressed = -999.0
        self._is_key_pressed = False
        self._KEY_QUIT = 'Q'
        self._is_key_quit_pressed = False

        # image size
        self._H1 = h1
        self._H2 = h2
        self._W1 = w1
        self._W2 = w2

        # Set three sub images for display
        self.set_img1()
        self.set_img2()
        self.set_img3()

        # Start the thread for displaying image
        IS_DEBUG = False
        if IS_DEBUG:
            self._thread_display_image()  # No extra thread. Easier to debug.
        else:
            self._thread = threading.Thread(target=self._thread_display_image,
                                            args=())
            self._thread.daemon = True
            self._thread.start()

    def is_key_pressed(self):
        return self._is_key_pressed

    def is_key_released(self):
        return not self._is_key_pressed

    def is_key_quit_pressed(self):
        return self._is_key_quit_pressed

    def _thread_display_image(self):
        ''' The thread for dislaying GUI.
            This thread is started when the class is initialized.
        '''
        while True:
            img_disp = np.hstack((np.vstack(
                (self._img1, self._img2)), self._img3))
            cv2.imshow("Audio classification", img_disp)
            q = cv2.waitKey(30)

            if time.time() - self._t_prev_pressed < self._KEY_EFFECTIVE_DURATION:
                # Key is considered as pressed. Nothing needs to change.
                pass
            else:  # Update key state
                self._result_key = chr(q).upper() if q > 0 else ''
                self._is_key_pressed = self._result_key == self._HOTKEY
                if self._is_key_pressed:  # Udpate time
                    self._t_prev_pressed = time.time()
                if self._result_key == self._KEY_QUIT:
                    self._is_key_quit_pressed = True

    def _init_blank_img(self, height, width, board_width=3):
        img = 255 + np.zeros((height, width), np.uint8)
        add_black_border(img, board_width)
        return img

    def set_img1(
            self,
            recording_length=0.0,
            voice_intensity=0.0,
    ):
        '''
        Image for displaying current recording state.
        Example:
        -----------------------------------------------
        Press `R` to record:
        Recording state: [============================]
        Voice intensity: [============================]    
        -----------------------------------------------
        '''

        # Configurations
        h, w = self._H1, self._W1
        img = self._init_blank_img(h, w)
        max_recording_length = 2.0  # unit: seconds
        max_voice_intensity = 1.0

        # Bars to show progress
        #   Recording state: [============================]
        #   Voice intensity: [============================]
        def create_bar(cur_val, max_val, n_symbols=20):
            ratio = min(1.0, abs(cur_val) / max_val)
            n1 = int(n_symbols * ratio)
            n2 = n_symbols - n1
            return "=" * n1 + " " * n2

        bar1 = create_bar(recording_length, max_recording_length)
        bar2 = create_bar(voice_intensity, max_voice_intensity)

        # Draw texts
        s1 = "Press 'R' to record:"
        s2 = "Recording state: [" + bar1 + "]"
        s3 = "Voice intensity:  [" + bar2 + "]"
        box = TextBoxDrawer(img, x0=20, y0=20, texts=[s1, s2, s3])

        # Return
        self._img1 = img

    def set_img2(
            self,
            final_label="None",
            predicted_label="",
            probability=0,
            length=0,
            valid_length=0,
    ):
        '''
        Image for displaying audio classification result.
        Example:
        -----------------------------------------------
        Final label:        walk
        Predicted label:    walk
        Probability:        99 %
        Length: 	        1.2 s
        Valid length:       0.9 s
        Last update:	    10:23
        -----------------------------------------------
        '''

        # Configurations
        h, w = self._H2, self._W1
        img = self._init_blank_img(h, w)

        # Draw texts
        s1a = "                    "
        s1 = "Final label:        "
        s1b = "                    "
        s2 = "Predicted label:    " + predicted_label
        s3 = "Probability:        {:.1f} %".format(probability * 100)
        s4 = "Length:            {:.1f} s".format(valid_length)
        s5 = "Valid length:       {:.1f} s".format(valid_length)
        s6 = "Last update:       " + get_hour_and_minute_and_seconds()
        TextBoxDrawer(img, x0=20, y0=20,
                      texts=[s1a, s1, s1b, s2, s3, s4, s5, s6])

        # Draw final label with a larger font
        cv2.putText(img, final_label, (240, 90),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=2.0, color=(0, 0, 0),
                    thickness=2)

        # Return
        self._img2 = img

    def set_img3(self, probabilities=[]):
        '''
        Image for displaying the probability of audio classification result.
        Example:
        -----------------------------------------------
        walk:   0.120
        run:    0.401
        kick:   0.133
        jump:   0.382
        etc.
        -----------------------------------------------
        '''

        # Check input
        N = len(self._CLASSES)  # number of classes
        assert (not probabilities or len(probabilities) == N)
        if not probabilities:
            probabilities = [0.0] * N

        # Configurations
        h, w = self._H1 + self._H2, self._W2
        img = self._init_blank_img(h, w)

        # Draw texts
        cv2.putText(img, "Probabilities:", (20, 50),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.1, color=(0, 0, 0),
                    thickness=1)
        y0, fontsize = 60, 0.7
        col1 = TextBoxDrawer(img, x0=20, y0=y0,
                             fontsize=fontsize, texts=self._CLASSES)
        col2 = TextBoxDrawer(img, x0=150, y0=y0,
                             fontsize=fontsize)
        for prob in probabilities:
            col2.add_text(":{:>6.1f}%".format(prob))

        # Return
        self._img3 = img


def read_list(filename):
    with open(filename) as f:
        with open(filename, 'r') as f:
            data = [l.rstrip() for l in f.readlines()]
    return data


def test_GUI():
    PATH_TO_CLASSES = ROOT + "config/classes.names"
    classes = read_list(PATH_TO_CLASSES)
    gui = GuiForAudioClassification(classes)
    while True:
        print(gui.is_key_pressed(), gui._result_key)
        time.sleep(0.1)


if __name__ == '__main__':
    test_GUI()
