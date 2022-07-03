# Drowsiness Detection System
This is a project implementing Computer Vision and Deep Learning concepts to detect drowsiness of a driver and sound an alarm if drowsy

### Description
In this deep learning model which first detect the face and eyes and based on the status of the eyes if the eyes are closed more than usual time then it can generate an alarm. This can be used by riders who tend to drive for a longer period of time that may lead to accidents.

### Technologies used
Python, OpenCV, Tensorflow, Keras

### Code Requirements
Python 3.7.6
OpenCv 4.5.2

### Libraries
import cv2
import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

### Model
1.eading all the images from the dataset and converting them into an array for Data & Labels.
2.Random shuffle is used to minimize overfitting
3.Using TensorFlow, import a Keras image classification model, optionally loaded with weights pre-trained on ImageNet. model = tf.keras.applications.mobilenet.MobileNet()
4.Use Tranfer Learning to create new model from above pretrained model
5.A single unit dense layer is acting as output for binary classification with activation set to Sigmoid
6.Optimizer is set to Adam.
7.For realtime time eyetracking in videos, use Haar Cascades frontal face algo.
