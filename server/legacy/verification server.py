import torch
import numpy as np
import os
import asyncio
import json
import websockets
from io import BytesIO

from PIL import Image, ImageDraw
from IPython import display

from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

model = InceptionResnetV1().eval().to(device)

async def get_embeddings(face_list):
  global model
  x = torch.Tensor(face_list).to(device)
  yhat = model(x)
  return yhat

def get_distance(someone, database):
  distance = [(someone - data).norm().item() for data in database]
  return distance

def get_argmin(someone, database):
  distance = get_distance(someone, database)
  for i in range(len(distance)):
    return np.argmin(distance)
  return -1

async def recv_face(websocket, path):
    buf = await websocket.recv()
    face = np.frombuffer(buf, dtype = np.float32)
    face = face.reshape((1,3,160,160))
    remote_ip = websocket.remote_address[0]
    msg='[{ip}] receive face properly, numpy shape={shape}'.format(ip=remote_ip, shape=face.shape)
    print(msg)
    embedding = await get_embeddings(face)
    await websocket.send('100')
    ##embedding DB서버에 넘기기##

print('run verification server')
start_server = websockets.serve(recv_face, '0.0.0.0', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()