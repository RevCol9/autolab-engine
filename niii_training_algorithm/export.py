import threading
import cv2
from ultralytics import YOLO

model=YOLO('yolov8n.pt')
model.export(format='engine')