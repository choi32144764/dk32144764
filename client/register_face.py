import tkinter as tk
import tkinter.font
import tkinter.messagebox
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
class Register(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.uri = ''

        self.device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=device)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam_width = 640
        self.cam_height = 480
        self.cap.set(3, self.cam_width)
        self.cap.set(4, self.cam_height)

        self.detecting_square = (200, 200)        
        self.detected = False
        self.face_list = []
        self.image_list = []

        self.width = 740
        self.height = 640
        self.parent = parent
        self.parent.title("출석 데이터 등록")
        self.parent.geometry("%dx%d+100+100" % (self.width, self.height))
        self.pack()
        self.create_widgets()
        # Event loop and Thread
        # self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.mainthread)
        self.thread.start()
    def create_widgets(self):
        image = np.zeros([self.cam_height,self.cam_width,3], dtype=np.uint8)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        font = tk.font.Font(family="맑은 고딕", size=15)
        
        self.alert = tk.Label(self, text="카메라를 정면으로 향하고 화면의 사각형에 얼굴을 맞춰주세요", font=font)
        self.alert.grid(row=0, column=0, columnspan=20)
        self.label = tk.Label(self, image=image)
        self.label.grid(row=1, column=0, columnspan=20)
        self.studentID = tk.StringVar()
        self.studentIdLabel = tk.Label(self, text="학번")
        self.studentIdLabel.grid(row=2, column=10)
        self.studentIdEntry = tk.Entry(self, width=20, textvariable=self.studentID)
        self.studentIdEntry.grid(row=2, column=11)
        
        self.studentName = tk.StringVar()
        self.studentNameLabel = tk.Label(self, text="이름")
        self.studentNameLabel.grid(row=3, column=10)
        self.studentNameEntry = tk.Entry(self, width=20, textvariable=self.studentName)
        self.studentNameEntry.grid(row=3, column=11)
        self.registerButton = tk.Button(self, text="등록", fg="blue", command=self.register_face)
        self.registerButton.grid(row=4, column=10)
        self.registerButton = tk.Button(self, text="다시촬영", command=self.restart)
        self.registerButton.grid(row=4, column=11)
        
        self.quit = tk.Button(self, text="나가기", fg="red", command=self.stop)
        self.quit.grid(row=5, column=10)
    def register_face(self):
        if not self.detected:
            tk.messagebox.showinfo("경고", "얼굴이 인식되지 않았습니다.")
            return
        asyncio.get_event_loop().run_until_complete(self.send_face())
            
    def restart(self):
        if not self.thread.isAlive():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(3, self.cam_width)
            self.cap.set(4, self.cam_height)
            
            self.detected = False
            self.face_list = []
            self.image_list = []
            
            self.thread = threading.Thread(target=self.mainthread)
            self.thread.start()
    
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
                self.alert.config(text= "인식된 얼굴이 너무 작습니다. 카메라에 더 가까이 접근해주세요.", fg="red")
                self.alert.update()
                continue
            image = frame
            image_list.append(image)
            # tensor 데이터 저장
            face_list.append(face.numpy())
        return face_list, image_list
    def mainthread(self):
        t = threading.currentThread()
        x1 = int(self.cam_width / 2 - self.detecting_square[0] / 2)
        x2 = int(self.cam_width / 2 + self.detecting_square[0] / 2)
        y1 = int(self.cam_height / 2 - self.detecting_square[1] / 2)
        y2 = int(self.cam_height / 2 + self.detecting_square[1] / 2)
        detected_time = None
        while getattr(t, "do_run", True):
            ret, frame = self.cap.read()
            # model에 이용하기 위해 convert
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 사각형 영역만 검사
            face_list, image_list = self.detect_face(converted[y1:y2, x1:x2])
            # 얼굴이 인식된 경우 파란색 사각형을 띄움
            if face_list:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # BGR color에서 RGB로 변환
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 유저에게 보여줄 땐 거울상으로 보여준다
            converted = cv2.flip(converted,1)
            image = Image.fromarray(converted)
            image = ImageTk.PhotoImage(image)
            self.label.configure(image=image)
            self.label.image = image # kind of double buffering
            # 얼굴이 인식되면 멤버함수에 넣음
            if face_list:
                self.face_list = face_list
                self.image_list = image_list
                # 2초 후에 사진이 찍힘
                if detected_time is None:
                    detected_time = time.time()
                else:
                    self.alert.config(text= "얼굴이 인식되었습니다. %f초 후 사진을 촬영합니다"%(2-(time.time()-detected_time)), fg="red")
                    if time.time() - detected_time >= 2:
                        self.thread.do_run = False
                        self.detected = True
                        self.alert.config(text= "얼굴을 등록해주세요. 올바르게 촬영되지 않았을 경우 다시촬영을 눌러주세요.", fg="blue")
            else:
                detected_time = None
                self.face_list = []
                self.image_list = []
                
            
    async def wait(self, n):
        await asyncio.sleep(n)
        
    async def send_face(self):
        try:
            async with websockets.connect(self.uri) as websocket:
                for face, image in zip(self.face_list, self.image_list):
                    #type: np.float32
                    send = json.dumps({'action': 'register',
                                       'student_id':self.studentID.get(),
                                       'student_name':self.studentName.get(),
                                       'tensor': face.tolist()})
                    await websocket.send(send)
                    recv = await websocket.recv()
                    data = json.loads(recv)
                    if data['status'] == 'success':
                        tk.messagebox.showinfo("등록완료", self.studentID.get() + ' ' + self.studentName.get())
        except Exception as e:
            tk.messagebox.showinfo("등록실패", e)
    def stop(self):
        self.thread.do_run = False
        # self.thread.join() # there is a freeze problem
        # self.event_loop.close()
        self.cap.release()
        self.parent.destroy()
        
if __name__ == '__main__':
    root = tk.Tk()
    Register(root)
    root.mainloop()
