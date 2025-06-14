import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from concurrent.futures import ThreadPoolExecutor
import queue
import json
import argparse
import warnings
from datetime import datetime
import os
import sys

# 경고 억제
def suppress_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
suppress_warnings()

parser = argparse.ArgumentParser(description="YOLO + MiDaS Pose Detection")
parser.add_argument('--camera_url', type=str, help='RTSP or video path', default=0)
parser.add_argument('--message_num', type=int, help='Message number from server', default=1)
args = parser.parse_args()

#cam 스위
#video_path = './VIRAT_S_010204_05_000856_000890.mp4'
video_path = 0
message_num = 1

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
fps = 8
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = 'output_depth_pose.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

model = YOLO('yolov8m-pose.pt')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

frame_queue = queue.Queue(maxsize=10)
executor = ThreadPoolExecutor(max_workers=4)

class CustomAnnotator(Annotator):
    def __init__(self, image, line_width=3, font_size=None, font="Arial.ttf", pil=False):
        super().__init__(image, line_width=line_width, font_size=font_size, font=font, pil=pil)
    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        super().box_label(box, label=label, color=color, txt_color=txt_color)
    def kpts(self, kpts, shape=(640, 640), radius=None, kpt_line=True, conf_thres=0.25, kpt_color=None):
        super().kpts(kpts)

def frame_reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)  # None을 큐에 넣어 종료 신호
            frame_queue.put(None)
            frame_queue.put(None)
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.01)
executor.submit(frame_reader)

def process_midas(img):
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img.shape[:2], mode='bicubic', align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return 1.0 - depth_map_normalized

def process_yolo(image):
    results = list(model.predict(image, stream=True, verbose=False, conf=0.5))
    return results

def picture_in_picture(main_image, overlay_image, img_ratio=5, border_size=3, position="left"):
    new_height = main_image.shape[0] // img_ratio
    new_width = int(new_height * (overlay_image.shape[1] / overlay_image.shape[0]))
    overlay_resized = cv2.resize(overlay_image, (new_width, new_height))
    overlay_with_border = cv2.copyMakeBorder(
        overlay_resized, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    x_margin = 10 if position == "left" else main_image.shape[1] - overlay_with_border.shape[1] - 10
    y_offset = 50
    end_y = y_offset + overlay_with_border.shape[0]
    end_x = x_margin + overlay_with_border.shape[1]
    if end_y > main_image.shape[0] or end_x > main_image.shape[1]:
        return main_image
    main_image[y_offset:end_y, x_margin:end_x] = overlay_with_border
    return main_image

def create_detection_data(bbox, cls_name, confidence):
    x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
    w = x_max - x_min
    h = y_max - y_min
    return {
        "EquipmentType": "AIBox", "EquipmentID": 1, "MessageType": "DetectionReport",
        "SensorType": "Fusion", "TrackID": 1, "TargetType": cls_name, "TargetNumber": 1,
        "H": h, "W": w, "X": x_min, "Y": y_min, "Speed": 5.5,
        "Probability": round(confidence, 3), "MessageNum": message_num
    }

# def stabilize_frame(prev_gray, curr_frame):
#     curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
#     prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
#     curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
#     idx = np.where(status == 1)[0]
#     prev_pts = prev_pts[idx]
#     curr_pts = curr_pts[idx]
#     m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
#     stabilized = cv2.warpAffine(curr_frame, m, (curr_frame.shape[1], curr_frame.shape[0]))
#     return stabilized, curr_gray

def stabilize_frame(prev_gray, curr_frame, zoom_ratio=1.05):
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    #머신러닝 기반.
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    stabilized = cv2.warpAffine(curr_frame, m, (curr_frame.shape[1], curr_frame.shape[0]))

    # 줌인 효과 적용 (중앙 crop 후 확대)
    h, w = stabilized.shape[:2]
    crop_margin_x = int((1 - 1/zoom_ratio) * w / 2)
    crop_margin_y = int((1 - 1/zoom_ratio) * h / 2)
    cropped = stabilized[crop_margin_y:h - crop_margin_y, crop_margin_x:w - crop_margin_x]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed, curr_gray

prev_gray = None
skip_frames = 3
frame_count = 0
start_time = time.time()
exit_flag = False

try:
    while not exit_flag:
        detection_results = []
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                print("[INFO] 영상 끝. 자동 종료.")
                exit_flag = True
                break
        except queue.Empty:
            continue

        if prev_gray is None:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue
        else:
            frame, prev_gray = stabilize_frame(prev_gray, frame)

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        img = frame.copy()
        midas_future = executor.submit(process_midas, img)
        yolo_future = executor.submit(process_yolo, img)
        depth_map = midas_future.result()
        results = yolo_future.result()

        for predictions in results:
            custom_annotator = CustomAnnotator(predictions.orig_img)
            if predictions is None or predictions.boxes is None or predictions.keypoints is None:
                continue
            for bbox, keypoints in zip(predictions.boxes, predictions.keypoints):
                for conf, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                    xmin, ymin, xmax, ymax = map(int, bbox_coords)
                    confidence = float(conf)
                    cls_name = "person" if int(classes) == 0 else None
                    color = colors(int(classes))
                    custom_annotator.box_label(bbox_coords, label=None, color=tuple(reversed(color)))
                    custom_annotator.kpts(keypoints.data[0])
                    detection_data = create_detection_data(bbox, cls_name, confidence)
                    detection_results.append(detection_data)
                    distance = None
                    if keypoints is not None and len(keypoints.xy[0]) > 0:
                        kp = keypoints.xy[0][0]
                        x, y = map(int, kp[:2])
                        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                            distance = depth_map[y, x] * 9
                    text_lines = [
                        f"{cls_name if cls_name else 'Unknown'} {confidence * 100:.1f}%",
                        f"Dist: {distance:.2f} m" if distance else "Dist: N/A"
                    ]
                    font_scale = 0.5
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_color = (255, 255, 255)
                    bg_color = (0, 0, 0)
                    line_spacing = 6
                    text_sizes = [cv2.getTextSize(line, font, font_scale, 1)[0] for line in text_lines]
                    max_width = max(w for w, h in text_sizes)
                    total_height = sum(h for w, h in text_sizes) + (len(text_lines) - 1) * line_spacing
                    text_x = xmin
                    text_y = ymin - total_height - 5 if ymin - total_height - 5 > 0 else ymax + 5
                    cv2.rectangle(img, (text_x, text_y - 20), (text_x + max_width + 10, text_y + total_height), bg_color, thickness=cv2.FILLED)
                    for i, line in enumerate(text_lines):
                        line_y = text_y + i * (text_sizes[i][1] + line_spacing)
                        cv2.putText(img, line, (text_x + 5, line_y), font, font_scale, font_color, 1)

        annotated_frame = custom_annotator.result()
        depth_map_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        frame = picture_in_picture(annotated_frame, depth_map_colored)
        end_time = time.time()
        fps_display = f"FPS: {10 / (end_time - start_time):.2f}"
        start_time = time.time()
        cv2.putText(frame, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('Pose Detection with Depth', frame)
        if detection_results:
            print(json.dumps(detection_results), flush=True)
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = os.path.join('detection_result', f"detected_frame_{current_time}.jpg")
            cv2.imwrite(file_name, frame)
            sys.stdout.flush()
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("[INFO] 사용자 종료.")
            exit_flag = True
finally:
    if cap.isOpened():
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)
