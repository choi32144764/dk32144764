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

                # DB에 연결
                cursor = attendance_db.cursor(pymysql.cursors.DictCursor)

                # 학생을 찾음
                sql = "SELECT student_id FROM student WHERE student_id = %s;"
                rows_count = cursor.execute(sql, (student_id))

                # DB에 학생이 없으면 등록
                if rows_count == 0:
                    sql = "INSERT INTO student(student_id, student_name) VALUES (%s, %s)"
                    cursor.execute(sql, (student_id, student_name))
                    sql = "INSERT INTO lecture_students(lecture_id, student_id) VALUES (%s, %s)"
                    cursor.execute(sql, ('0', student_id))
                    msg='[{ip}] {id} is registered'.format(ip=remote_ip, id=student_id)
                    print(msg)

                # student_embedding Table에 등록
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

                # 가장 비슷한 Embedding을 찾는 SQL
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

                # 출석 데이터 전송
                send = ''
                if distance_min < 0.62:
                    # 인증 성공
                    # 오늘 이미 출석 됐는지 확인
                    sql = "SELECT DATE(timestamp) FROM student_attendance WHERE (lecture_id=%s) AND (student_id=%s) AND (DATE(timestamp) = CURDATE());"
                    rows_count = cursor.execute(sql, ('0', verified_id))
                    
                    # 출석 기록이 없는 경우에만
                    if rows_count == 0:
                        # 테이블 맨 뒤에 datetime attribute가 있음. 서버 시간 가져오게 default로 설정해둠.
                        sql = "INSERT INTO student_attendance(lecture_id, student_id, status) VALUES (%s, %s, %s)"
                        # TODO: attend / late 처리
                        cursor.execute(sql, ('0', verified_id, 'attend'))
                        attendance_db.commit()
                        # log 작성
                        msg='[{ip}] verification success {id}'.format(ip=remote_ip, id=verified_id)
                        print(msg)
                        send = json.dumps({'status': 'success', 'student_id': verified_id})
                    else:
                        msg='[{ip}] verification failed: {id} is already verified'.format(ip=remote_ip, id=verified_id)
                        print(msg)
                        send = json.dumps({'status': 'already', 'student_id': verified_id})
                else:
                    # 인증 실패
                    msg='[{ip}] verification failed'.format(ip=remote_ip)
                    print(msg)
                    send = json.dumps({'status': 'fail'})                    
                await websocket.send(send)
            elif data['action'] == "save_image":
                # 출석이 제대로 이뤄지지 않으면 이미지를 저장하여
                # 나중에 교강사가 출석을 확인할 수 있도록 한다
                msg='[{ip}] save image'.format(ip=remote_ip)
                print(msg)
                arr = np.asarray(data['image'], dtype = np.uint8)
                blob = arr.tobytes()
                # 테이블 맨 뒤에 datetime attribute가 있음. 서버 시간 가져오게 default로 설정해둠.
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
