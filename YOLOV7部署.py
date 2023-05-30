import cv2
import json
import serial
import threading
import torch
import torch.backends.cudnn as cudnn
import onnxruntime
import numpy as np

cap = cv2.VideoCapture(0)

labels = ['t1', 't2', 't3', 't4', 't5']
num_classes = len(labels)
signals = {'t1': '花腹鯖', 't2': '腳踏車', 't3': '汽車', 't4': '摩托車'}

ort_session = onnxruntime.InferenceSession('C:\\Users\\tfr52\\OneDrive\\桌面\\yolov7\\best.onnx')
input_name = ort_session.get_inputs()[0].name

def Fish_Json(detections):
    detections_json = json.dumps({'detections': detections})
    print(detections_json)
    ser = serial.Serial('COM3', 9600, timeout=1)
    response = ser.write(detections_json.encode())
    print("response", response)
    ser.close()

def detect_and_upload():
    while True:
        ret, frame = cap.read()  # 在迴圈開始前讀取一個幀
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        rgb = np.transpose(rgb, (2, 0, 1))  # 轉換維度順序，形狀變為 (3, 640, 640)
        rgb = np.array(rgb, dtype=np.float32)
        rgb /= 255.0
        rgb = np.expand_dims(rgb, axis=0)  # 擴展維度，形狀變為 (1, 3, 640, 640)
        results = ort_session.run(None, {input_name: rgb})[0]
        detections = []
        for obj in results:
            class_id = int(obj[5])
            if class_id >= 0 and class_id < num_classes:
                label = labels[class_id]
                if label in signals:
                    signal = signals[label]
                    print('檢測到' + signal)
                    detections.append(signal)
            else:
                continue
        if detections:
            Fish_Json(detections)
        cv2.imshow('test', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    t = threading.Thread(target=detect_and_upload)
    t.start()
    if cap.isOpened():
        cap
