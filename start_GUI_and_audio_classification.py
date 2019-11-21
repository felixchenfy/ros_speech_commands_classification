#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

if 1:  # Set path
    import sys
    import os
    # Root of the project
    ROOT = os.path.dirname(os.path.abspath(__file__)) + "/"
    sys.path.append(ROOT)
    
import rospy
import time
from std_msgs.msg import Float32
from std_msgs.msg import String
from src.s4_inference_microphone_by_GUI import AudioClassifierWithGUI


# -- Settings
SRC_WEIGHT_PATH = ROOT + "weights/my.ckpt"
SRC_CLASSES_PATH = ROOT + "config/classes.names"

DST_AUDIO_FOLDER = ROOT + "data/data_tmp/"

ROS_TOPIC_PREDICTED_LABEL = "ros_speech_commands_classification/predicted_label"
ROS_TOPIC_PREDICTED_PROBABILITY = "ros_speech_commands_classification/predicted_probability"


# -- Main


class Publishers(object):
    ''' ROS publisher for predicted label and probability '''

    def __init__(self):
        self._pub_label = rospy.Publisher(
            ROS_TOPIC_PREDICTED_LABEL,
            String, queue_size=10)
        self._pub_prob = rospy.Publisher(
            ROS_TOPIC_PREDICTED_PROBABILITY,
            Float32, queue_size=10)

    def publish(self, label, prob):
        self._pub_label.publish(label)
        self._pub_prob.publish(prob)

class TimerPrinter(object):
    ''' A class for printing message with time control:
        If previous `print` is within T_gap seconds, 
        then the currrent `print` prints nothing. 
    '''

    def __init__(self, print_period):
        self._prev_time = -999.0
        self._t_print_gap = print_period

    def print(self, s):
        if time.time() - self._prev_time >= self._t_print_gap:
            self._prev_time = time.time()
            print(s)

def inference_from_microphone():

    # Audio recorder and classifier
    audio_clf = AudioClassifierWithGUI(
        SRC_WEIGHT_PATH, SRC_CLASSES_PATH, DST_AUDIO_FOLDER)

    # ROS publishers
    publishers = Publishers()

    # Start loop
    timer_printer = TimerPrinter(print_period=2.0)  # for print
    while (not rospy.is_shutdown()) and \
            not audio_clf.is_key_quit_pressed():
        timer_printer.print("Usage: keep pressing down 'R' to record audio")
        if audio_clf.is_key_pressed():
            label, prob = audio_clf.record_audio_and_classifiy()
            publishers.publish(label, prob)
        rospy.sleep(0.001)


def test_publishers():
    publishers = Publishers()
    label, prob = "dog", 1.2
    while not rospy.is_shutdown():
        publishers.publish(label, prob)
        print("Published: ", label, prob)
        rospy.sleep(1.0)
        prob += 0.1

if __name__ == "__main__":
    rospy.init_node("start_GUI_and_audio_classification")
    inference_from_microphone()
    # test_publishers()
