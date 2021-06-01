import numpy as np
import cv2
import asyncio
import websockets
from io import BytesIO

from PIL import Image, ImageDraw
from IPython import display

async def recv_image(websocket, path):
    buf = await websocket.recv()
    byte = BytesIO(buf)
    image = Image.open(byte)
    remote_ip = websocket.remote_address[0]
    msg='[{ip}] receive face properly, image size={size}'.format(ip=remote_ip, size=image.size)
    print(msg)
    await websocket.send('100')
    #for debug
    #frame = np.array(image)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #cv2.imshow('recv', frame)
    #cv2.waitKey(2000)
    #cv2.destroyAllWindows()

print('run image server')
start_image_server = websockets.serve(recv_image, '0.0.0.0', 8766)
asyncio.get_event_loop().run_until_complete(start_image_server)
asyncio.get_event_loop().run_forever()