This project is written in Python 2.7 on Ubuntu 18.04. 

The main depencencies are listed below.  

## All python packages  

```
$ sudo pip2 install matplotlib sklearn scipy numpy opencv-python jupyter future-fstrings
$ sudo pip2 install pynput soundfile sounddevice librosa gtts pyttsx pyttsx3  
```

Then install Pytorch:    
Please go to https://pytorch.org/ and install the one that matches with your computer.  
I'm using the version: Stable(1.3), Linux, Pip, Python 2.7, CUDA 10.1:  
```
$ sudo pip install torch torchvision # or pip2
```

## Descriptions about the packages 
* Keyboard io:  
    ```
    $ sudo pip2 install pynput  
    ```

* Audio IO:  
    https://github.com/bastibe/SoundFile  
    ```
    $ sudo pip2 install soundfile  
    $ sudo pip2 install sounddevice  
    $ sudo apt-get install libsndfile1  
    ```

* Extract audio MFCC feature:  
    https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html  
    ```
    $ sudo pip2 install librosa  
    ```

* Synthesize audio:  
    ```
    $ sudo pip2 install gtts  
    ```

* Synthesize audio (Tested but not used):  
    ```
    $ sudo pip2 install pyttsx  
    $ sudo pip2 install pyttsx3  
    $ sudo apt-get install espeak  
    ```
