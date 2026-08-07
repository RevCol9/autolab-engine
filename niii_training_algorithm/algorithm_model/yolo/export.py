import threading
import cv2
from ultralytics import YOLO

model=YOLO('best.pt')
model.export(format='onnx')