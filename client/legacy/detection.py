import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageDraw
from IPython import display

from models import mtcnn
from models import inception_resnet_v1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def extract_face(filename, required_size=(224, 224)):
  # If required, create a face detection pipeline using MTCNN:
  mtcnn_model = mtcnn.MTCNN(keep_all=True, device=device)
  pixels = plt.imread(os.path.join(os.path.abspath(''), filename))
  results = mtcnn_model.detect(pixels)
  face_array = []
  for box, prob in zip(results[0], results[1]):
    #boxes, _ = result
    print('face detected. prob:', prob)
    x1, y1, x2, y2 = box
    face = pixels[int(y1):int(y2), int(x1):int(x2)]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array.append(np.asarray(image))
  return face_array

face_array = extract_face('image/test1.jpg')
for face in face_array:
  plt.figure()
  plt.imshow(face)
  plt.show()

face_array = extract_face('image/test2.jpg')
for face in face_array:
  plt.figure()
  plt.imshow(face)
  plt.show()

