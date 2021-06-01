##################################################
#1. webcam에서 얼굴을 인식합니다.                                           #
#2. 얼굴일 확률이 95% 이상인 이미지를 이미지 서버로 전송합니다.  #
#3. 전처리 된 데이터를 verification 서버에 전송합니다.                   #
##################################################
import torch
import numpy as np
import cv2
import asyncio
import websockets
import json
import os
import timeit
import base64

from PIL import Image
from io import BytesIO
import requests

from models.mtcnn import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

uri = 'ws://localhost:8765'

async def send_face(face_list, image_list):
    global uri
    async with websockets.connect(uri) as websocket:
        for face, image in zip(face_list, image_list):
            #type: np.float32
            send = json.dumps({"action": "verify", "MTCNN": face.tolist()})
            await websocket.send(send)
            recv = await websocket.recv()
            data = json.loads(recv)
            if data['status'] == 'success':
                # 성공
                print(data['id'], 'is attend')
            else:
                print('verification failed')
                send = json.dumps({'action': 'save_image', 'image': image.tolist(), 'shape': image.shape})
                await websocket.send(send)

async def send_image(image_list):
    global uri
    async with websockets.connect(uri) as websocket:
        for image in image_list:
            data = json.dumps({'action': 'save_image', 'image': image.tolist(), 'shape': image.shape})
            await websocket.send(data)
        print('send', len(image_list), 'image(s)')
        code = await websocket.recv()
        print('code:', code)

def detect_face(frame):
    # If required, create a face detection pipeline using MTCNN:
    global mtcnn
    results = mtcnn.detect(frame)
    image_list = []
    if results[1][0] == None:
        return []
    for box, prob in zip(results[0], results[1]):
        if prob < 0.95:
            continue
        print('face detected. prob:', prob)
        x1, y1, x2, y2 = box
        image = frame[int(y1-10):int(y2+10), int(x1-10):int(x2+10)]
        image_list.append(image)
    return image_list

def make_face_list(frame):
    global mtcnn
    results, prob = mtcnn(frame, return_prob = True)
    face_list = []
    if prob[0] == None:
        return []
    for result, prob in zip(results, prob):
        if prob < 0.95:
            continue
        #np.float32
        face_list.append(result.numpy())
    return face_list

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)
while True:
    try:
        #start = timeit.default_timer()
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_list = make_face_list(frame)
        image_list = detect_face(frame)
        ##embedding server로 전송##
        if face_list:
            asyncio.get_event_loop().run_until_complete(send_face(face_list, image_list))
        ###################
        ##image server로 전송##
        #if image_list:
            #asyncio.get_event_loop().run_until_complete(send_image(image_list))
        ###################
        #end = timeit.default_timer()
        #print('delta time: ', end - start)
    except Exception as ex:
        print(ex)
