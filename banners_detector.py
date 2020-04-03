
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
import json
import zlib
import base64


from os import listdir
from xml.etree import ElementTree
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class myMaskRCNNConfig(Config):
	# give the configuration a recognizable name
	NAME = "MaskRCNN_config"

	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# number of classes (we would normally add +1 for the background)
	NUM_CLASSES = 1+8

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 1000
	VALIDATION_STEPS = 200

	# network
	BACKBONE = "resnet50"

	# Learning rate
	LEARNING_RATE=0.006

	# Skip detections with < 90% confidence
	DETECTION_MIN_CONFIDENCE = 0.9

	# setting Max ground truth instances
	MAX_GT_INSTANCES=10


if __name__ == '__main__':
	
	config = myMaskRCNNConfig()
	config.display()

	#Loading the model in the inference mode
	model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')

	model.load_weights('mask_rcnn_checkpoint.h5', by_name=True)


def detect_masks(frame, model=model):
    
	# convert frame for prediction
	frame_res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)

	# detecting objects in the image
	result = model.detect([frame_res])

	# bool type mask
	mask = result[0]['masks']

	# classes
	classes = result[0]['class_ids']

	# returning binary mask and classes
	return mask, contours[0]



