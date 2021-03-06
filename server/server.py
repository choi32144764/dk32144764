import os
import torch
import numpy as np
import asyncio
import json
import base64
import websockets
from io import BytesIO

import pymysql
from datetime import datetime

from PIL import Image, ImageDraw
from IPython import display

from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

model = InceptionResnetV1().eval().to(device)
attendance_db = pymysql.connect(
    user='root', 
    passwd='1234', 
    host='localhost', 
    db='attendance', 
    charset='utf8'
)

lock = asyncio.Lock()
clients = set()
#processes = []

async def get_embeddings(face_list):
  global model
  x = torch.Tensor(face_list).to(device)
  return model(x)

async def get_distance(arr1, arr2):
    distance = np.linalg.norm(arr1 - arr2)
    return distance

async def get_cosine_similarity(arr1, arr2):
    similarity = np.inner(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    return similarity

async def register(websocket):
    global lock
    global clients
    async with lock:
        clients.add(websocket)
        #remote_ip = websocket.remote_address[0]
        #msg='[{ip}] connected'.format(ip=remote_ip)
        #print(msg)

async def unregister(websocket):
    global lock
    global clients
    async with lock:
        clients.remove(websocket)
        #remote_ip = websocket.remote_address[0]
        #msg='[{ip}] disconnected'.format(ip=remote_ip)
        #print(msg)

async def thread(websocket, path):
    await register(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            remote_ip = websocket.remote_address[0]
            if data['action'] == 'register':
                # log
                msg='[{ip}] register face'.format(ip=remote_ip)
                print(msg)

                # load json
                student_id = data['student_id']
                student_name = data['student_name']
                face = np.asarray(data['tensor'], dtype = np.float32)
                face = face.reshape((1,3,160,160))

                # DB??? ??????
                cursor = attendance_db.cursor(pymysql.cursors.DictCursor)

                # ????????? ??????
                sql = "SELECT student_id FROM student WHERE student_id = %s;"
                rows_count = cursor.execute(sql, (student_id))

                # DB??? ????????? ????????? ??????
                if rows_count == 0:
                    sql = "INSERT INTO student(student_id, student_name) VALUES (%s, %s)"
                    cursor.execute(sql, (student_id, student_name))
                    sql = "INSERT INTO lecture_students(lecture_id, student_id) VALUES (%s, %s)"
                    cursor.execute(sql, ('0', student_id))
                    msg='[{ip}] {id} is registered'.format(ip=remote_ip, id=student_id)
                    print(msg)

                # student_embedding Table??? ??????
                embedding = await get_embeddings(face)
                embedding = embedding.detach().numpy().tobytes()
                embedding_date = datetime.now().strftime('%Y-%m-%d')
                sql = "insert into student_embedding(student_id, embedding_date, embedding) values (%s, %s, _binary %s)"
                cursor.execute(sql, (student_id, embedding_date, embedding))
                attendance_db.commit()
                send = json.dumps({'status': 'success', 'student_id': student_id})
                await websocket.send(send)

            elif data['action'] == 'verify':
                # log
                msg='[{ip}] verify face'.format(ip=remote_ip)
                print(msg)

                # load json
                face = np.asarray(data['tensor'], dtype = np.float32)
                face = face.reshape((1,3,160,160))

                embedding = await get_embeddings(face)
                embedding = embedding.detach().numpy()

                # ?????? ????????? Embedding??? ?????? SQL
                cursor = attendance_db.cursor(pymysql.cursors.DictCursor)
                sql = "SELECT student_id, embedding FROM student_embedding;"
                cursor.execute(sql)
                result = cursor.fetchall()
                verified_id = '0'
                distance_min = 99
                for row_data in result:
                    db_embedding = np.frombuffer(row_data['embedding'], dtype=np.float32)
                    db_embedding = db_embedding.reshape((1,512))
                    distance = await get_distance(embedding, db_embedding)
                    if (distance < distance_min):
                        verified_id = row_data['student_id']
                        distance_min = distance
                        break

                # ?????? ????????? ??????
                send = ''
                if distance_min < 0.62:
                    # ?????? ??????
                    # ?????? ?????? ?????? ????????? ??????
                    sql = "SELECT DATE(timestamp) FROM student_attendance WHERE (lecture_id=%s) AND (student_id=%s) AND (DATE(timestamp) = CURDATE());"
                    rows_count = cursor.execute(sql, ('0', verified_id))
                    
                    # ?????? ????????? ?????? ????????????
                    if rows_count == 0:
                        # ????????? ??? ?????? datetime attribute??? ??????. ?????? ?????? ???????????? default??? ????????????.
                        sql = "INSERT INTO student_attendance(lecture_id, student_id, status) VALUES (%s, %s, %s)"
                        # TODO: attend / late ??????
                        cursor.execute(sql, ('0', verified_id, 'attend'))
                        attendance_db.commit()
                        # log ??????
                        msg='[{ip}] verification success {id}'.format(ip=remote_ip, id=verified_id)
                        print(msg)
                        send = json.dumps({'status': 'success', 'student_id': verified_id})
                    else:
                        msg='[{ip}] verification failed: {id} is already verified'.format(ip=remote_ip, id=verified_id)
                        print(msg)
                        send = json.dumps({'status': 'already', 'student_id': verified_id})
                else:
                    # ?????? ??????
                    msg='[{ip}] verification failed'.format(ip=remote_ip)
                    print(msg)
                    send = json.dumps({'status': 'fail'})                    
                await websocket.send(send)
            elif data['action'] == "save_image":
                # ????????? ????????? ???????????? ????????? ???????????? ????????????
                # ????????? ???????????? ????????? ????????? ??? ????????? ??????
                msg='[{ip}] save image'.format(ip=remote_ip)
                print(msg)
                arr = np.asarray(data['image'], dtype = np.uint8)
                blob = arr.tobytes()
                # ????????? ??? ?????? datetime attribute??? ??????. ?????? ?????? ???????????? default??? ????????????.
                cursor = attendance_db.cursor(pymysql.cursors.DictCursor)
                sql = "INSERT INTO undefined_image(lecture_id, image, width, height) VALUES (%s, _binary %s, %s, %s)"
                cursor.execute(sql, ('0', blob, arr.shape[0], arr.shape[1]))
                attendance_db.commit()
            else:
                print("unsupported event: {}", data)
    finally:
        await unregister(websocket)

print('run verification server')
start_server = websockets.serve(thread, '0.0.0.0', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
