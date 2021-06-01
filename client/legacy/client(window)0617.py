##################################################
#1. webcam에서 얼굴을 인식합니다.
#2. 얼굴일 확률이 97% 이상이고 영역이 15000 이상인 이미지를 서버에 전송
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
import time

from PIL import Image
from io import BytesIO
import requests

from models.mtcnn import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, post_process=True, device=device)

uri = 'ws://169.56.95.131:8765'

async def send_face(face_list, image_list):
    async with websockets.connect(uri) as websocket:
        for face, image in zip(face_list, image_list):
            #type: np.float32
            send = json.dumps({'action': 'verify', 'MTCNN': face.tolist()})
            await websocket.send(send)
            recv = await websocket.recv()
            data = json.loads(recv)
            if data['status'] == 'success':
                # 성공
                print(data['student_id'], 'is attend')
            else:
                print('verification failed:', data['status'])
                if data['status'] == 'failed':
                    send = json.dumps({'action': 'save_image', 'image': image.tolist()})

def detect_face(frame):
    results = mtcnn.detect(frame)
    faces = mtcnn(frame, return_prob = False)
    image_list = []
    face_list = []
    if results[1][0] == None:
        return [], []
    for box, face, prob in zip(results[0], faces, results[1]):
        if prob < 0.97:
            continue
        print('face detected. prob:', prob)
        x1, y1, x2, y2 = box
        if (x2-x1) * (y2-y1) < 15000:
            # 얼굴 해상도가 너무 낮으면 무시
            continue
        # 얼굴 주변 ±3 영역  저장
        image = frame[int(y1-3):int(y2+3), int(x1-3):int(x2+3)]
        image_list.append(image)
        # MTCNN 데이터 저장
        face_list.append(face.numpy())
    return image_list, face_list

def make_face_list(frame):
    results, prob = mtcnn(frame, return_prob = True)
    face_list = []
    if prob[0] == None:
        return []
    for result, prob in zip(results, prob):
        if prob < 0.97:
            continue
        #np.float32
        face_list.append(result.numpy())
    return face_list

if __name__ == '__main__':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 720)
    cap.set(4, 480)
    #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    while True:
        try:
            ret, frame = cap.read()
            #cv2.imshow('img', frame)
            #cv2.waitKey(10)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_list, face_list = detect_face(frame)
            if not face_list:
                continue;
            asyncio.get_event_loop().run_until_complete(send_face(face_list, image_list))
            time.sleep(1)
        except Exception as ex:
            print(ex)
