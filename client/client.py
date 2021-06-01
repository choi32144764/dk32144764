/
#1. webcam에서 얼굴을 인식합니다.
#2. 얼굴일 확률이 97% 이상이고 영역이 15000 이상인 이미지를 서버에 전송
/
import tkinter as tk
import tkinter.font
import tkinter.messagebox
import tkinter.scrolledtext
import threading
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

from PIL import Image, ImageTk
from io import BytesIO
import requests

from models.mtcnn import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, post_process=True, device=device)

uri = 'ws://169.56.95.131:8765'

class Client(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        # URI
        self.uri = 'ws://169.56.95.131:8765'

        # Pytorch Model
        self.device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=device)

        # OpenCV
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam_width = 640
        self.cam_height = 480
        self.cap.set(3, self.cam_width)
        self.cap.set(4, self.cam_height)

        # Application Function
        
        # cam에서 MTCNN 적용하는 영역
        self.detecting_square = (500, 300)

        # 영상 위에 사각형 색상 지정
        self.rectangle_color = (0, 0, 255)
        
        # tkinter GUI
        self.width = 740
        self.height = 700
        self.parent = parent
        self.parent.title("출석시스템")
        self.parent.geometry("%dx%d+100+100" % (self.width, self.height))
        self.pack()
        self.create_widgets()
        
        # Event loop and Thread
        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.mainthread)
        self.thread.start()

    def create_widgets(self):
        image = np.zeros([self.cam_height, self.cam_width, 3], dtype=np.uint8)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        font = tk.font.Font(family="맑은 고딕", size=15)
        
        self.alert = tk.Label(self, text="출석시스템", font=font)
        self.alert.grid(row=0, column=0, columnspan=20)
        self.label = tk.Label(self, image=image)
        self.label.grid(row=1, column=0, columnspan=20)

        self.log = tk.scrolledtext.ScrolledText(self, wrap = tk.WORD, state=tk.DISABLED, width = 96, height = 10)
        self.log.grid(row=2, column=0, columnspan=20)

        
        self.quit = tk.Button(self, text="나가기", fg="red", command=self.stop)
        self.quit.grid(row=3, column=10)

    def logging(self, text):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tkinter.END, text)
        self.log.insert(tkinter.END, '\n')
        self.log.config(state=tk.DISABLED)
        
    
    def detect_face(self, frame):
        results = self.mtcnn.detect(frame)
        faces = self.mtcnn(frame, return_prob = False)
        image_list = []
        face_list = []
        if results[1][0] == None:
            return [], []
        for box, face, prob in zip(results[0], faces, results[1]):
            if prob < 0.97:
                continue
            # for debug
            # print('face detected. prob:', prob)
            x1, y1, x2, y2 = box
            if (x2-x1) * (y2-y1) < 15000:
                # 얼굴 해상도가 너무 낮으면 무시
                continue
            image = frame[int(y1):int(y2), int(x1):int(x2)]
            image_list.append(image)
            # tensor 데이터 저장
            face_list.append(face.numpy())
        return face_list, image_list

    def mainthread(self):
        t = threading.currentThread()
        asyncio.set_event_loop(self.event_loop)
        x1 = int(self.cam_width / 2 - self.detecting_square[0] / 2)
        x2 = int(self.cam_width / 2 + self.detecting_square[0] / 2)
        y1 = int(self.cam_height / 2 - self.detecting_square[1] / 2)
        y2 = int(self.cam_height / 2 + self.detecting_square[1] / 2)
        while getattr(t, "do_run", True):
            ret, frame = self.cap.read()
            # BGR to RGB
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_list, image_list = self.detect_face(converted[y1:y2, x1:x2])
            # 얼굴이 인식되면 출석요청
            if face_list:
                self.event_loop.run_until_complete(self.send_face(face_list, image_list))
            # 사각형 영역 표시
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), self.rectangle_color, 3)
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 거울상으로 보여준다
            converted = cv2.flip(converted,1)
            image = Image.fromarray(converted)
            image = ImageTk.PhotoImage(image)
            self.label.configure(image=image)
            self.label.image = image # kind of double buffering

    @asyncio.coroutine
    def set_rectangle(self):
        self.rectangle_color = (255, 0, 0)
        yield from asyncio.sleep(2)
        self.rectangle_color = (0, 0, 255)
        
    async def send_face(self, face_list, image_list):
        try:
            async with websockets.connect(uri) as websocket:
                for face, image in zip(face_list, image_list):
                    #type: np.float32
                    send = json.dumps({'action': 'verify', 'tensor': face.tolist()})
                    await websocket.send(send)
                    recv = await websocket.recv()
                    data = json.loads(recv)
                    if data['status'] == 'success':
                        # 성공
                        self.logging('출석확인: ' + data['student_id'])
                        asyncio.ensure_future(self.set_rectangle())
                    else:
                        # 이미지 DB에 저장, 일단 보류
                        #if data['status'] == 'fail':
                        #    send = json.dumps({'action': 'save_image', 'image': image.tolist()})
                        #    await websocket.send(send)
                        if data['status'] == 'already':
                            asyncio.ensure_future(self.set_rectangle())
        except Exception as e:
            self.logging(e)

    def stop(self):
        self.thread.do_run = False
        # self.thread.join() # there is a freeze problem
        self.event_loop.close()
        self.cap.release()
        self.parent.destroy()
        

if __name__ == '__main__':
    root = tk.Tk()
    Client(root)
    root.mainloop()
    
