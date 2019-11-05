This project is written in Python 2.7 on Ubuntu 18.04. 

The main depencencies are listed below.  

* General python packages:  
    > $ conda create -n py2 python=2.7
    > $ conda activate py2  
    > $ pip install matplotlib sklearn scipy numpy opencv-python jupyter future-fstrings
    > $ pip install pynput soundfile sounddevice librosa gtts pyttsx pyttsx3  

* Pytorch:  
    Please go to "https://pytorch.org/" and install the one that matches with your computer.  
    I'm using the version: Stable(1.3), Linux, Pip, Python 2.7, CUDA 10.1:  
    > $ pip install torch torchvision  

* Keyboard io:  
    > $ pip install pynput  

* Audio IO:  
    https://github.com/bastibe/SoundFile  
    > $ pip install soundfile  
    > $ pip install sounddevice  
    > $ sudo apt-get install libsndfile1  

* Extract audio MFCC feature:  
    https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html  
    > $ pip install librosa  

* Synthesize audio:  
    > $ pip install gtts  

* Synthesize audio (Tested but not used):  
    > $ pip install pyttsx  
    > $ pip install pyttsx3  
    > $ sudo apt-get install espeak  