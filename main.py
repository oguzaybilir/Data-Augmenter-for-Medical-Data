import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob 

paths = glob.glob("./images/*/*.jpg")


def data_augmenter(paths):

