from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO('yolov8m.pt')
# YOLO 예측 수행
results = model.predict(source=0, show=True)

