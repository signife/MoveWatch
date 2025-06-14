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


# RTSP 스트림 설정
# RTSP 스트림 설정 (TCP 클라이언트에서 전달된 camera_url 사용)
# 명령행 인자 설정
parser = argparse.ArgumentParser(description="YOLO + MiDaS Pose Detection")
parser.add_argument('--camera_url', type=str, help='RTSP or video path', default=0)
parser.add_argument('--message_num', type=int, help='Message number from server', default=1)
args = parser.parse_args()

#video_path = args.camera_url  # 명령행 인자로 전달된 camera_url 사용
#message_num = args.message_num
#print (video_path, message_num)
video_path = './VIRAT_S_010204_05_000856_000890.mp4'
#video_path = 'rtsp://admin:sensorway1@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif' # 3층
#video_path = 0
#video_path='rtsp://admin:sensorway1@192.168.202.153:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif' # 2층
message_num = 1


cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

#fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = 8
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = 'output_depth_pose.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# YOLO 모델 (포즈 탐지 모델)
model = YOLO('yolov8m-pose.pt')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# MiDaS 모델 로드 (깊이 추정)
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

# 프레임 읽기 스레드
def frame_reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.01)

executor.submit(frame_reader)

# MiDaS 처리 함수 (깊이 추정)
def process_midas(img):
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return 1.0 - depth_map_normalized

# YOLO 처리 함수 (포즈 탐지)
def process_yolo(image):
    results = list(model.predict(image, stream=True, verbose=False, conf=0.5))
    return results

def copy_frame(frame):
    """Copy the frame to avoid direct modification."""
    return frame.copy()

def picture_in_picture(main_image, overlay_image, img_ratio=5, border_size=3, position="left"):
    """
    프레임의 왼쪽 상단에 depth map을 자동으로 배치하도록 수정된 PIP 함수
    """
    # 오버레이 이미지 크기 조정
    new_height = main_image.shape[0] // img_ratio
    new_width = int(new_height * (overlay_image.shape[1] / overlay_image.shape[0]))
    overlay_resized = cv2.resize(overlay_image, (new_width, new_height))

    # 테두리 추가
    overlay_with_border = cv2.copyMakeBorder(
        overlay_resized,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    # 위치 조정
    if position == "left":
        # 왼쪽 상단
        x_margin = 10  # 항상 왼쪽에서 약간의 여백
        y_offset = 50  # FPS와 겹치지 않도록 아래로 이동
    elif position == "right":
        # 오른쪽 상단
        x_margin = main_image.shape[1] - overlay_with_border.shape[1] - 10  # 오른쪽에서 10 픽셀 여백
        y_offset = 10  # 위에서 약간의 여백


    # 삽입 가능 영역 확인
    end_y = y_offset + overlay_with_border.shape[0]
    end_x = x_margin + overlay_with_border.shape[1]

    if end_y > main_image.shape[0] or end_x > main_image.shape[1]:
        #print(f"Warning: Overlay image exceeds main image boundaries. Skipping overlay.")
        return main_image

    # 메인 이미지에 오버레이
    main_image[y_offset:end_y, x_margin:end_x] = overlay_with_border
    return main_image

def create_detection_data(bbox, cls_name, confidence):
    x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
    w = x_max - x_min
    h = y_max - y_min


    return {
        "EquipmentType": "AIBox",
        "EquipmentID": 1,
        "MessageType": "DetectionReport",
        "SensorType": "Fusion",
        "TrackID": 1,
        "TargetType": cls_name ,
        "TargetNumber": 1,
        "H": h,
        "W": w,
        "X": x_min,
        "Y": y_min,
        "Speed": 5.5,
        "Probability": round(confidence, 3),
        "MessageNum": message_num
    }

def save_detected_frame(frame, output_dir):
    """탐지된 프레임 저장"""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = os.path.join(output_dir, f"detected_frame_{current_time}.jpg")
    cv2.imwrite(file_name, frame)
    print(f"[저장 완료] 탐지된 프레임: {file_name}")

skip_frames = 3
frame_count = 0
start_time = time.time()

exit_flag = False
try:
    while not exit_flag:
        detection_results = []
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        img = frame.copy()

        # MiDaS와 YOLO 동시 실행
        midas_future = executor.submit(process_midas, img)
        yolo_future = executor.submit(process_yolo, img)
        overlay_frame_future = executor.submit(copy_frame, frame)

        depth_map = midas_future.result()
        results = yolo_future.result()
        overlay_frame = overlay_frame_future.result()

        # YOLO pose 모델 결과 그대로 시각화
        for predictions in results:
            custom_annotator = CustomAnnotator(predictions.orig_img)
            if predictions is None or predictions.boxes is None or predictions.keypoints is None:
                continue  # 탐지 결과가 없으면 다음 프레임으로 넘어감

            # 박스 내 텍스트 정보를 표시
            for bbox, keypoints in zip(predictions.boxes, predictions.keypoints):
                for conf, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                    xmin, ymin, xmax, ymax = map(int, bbox_coords)
                    confidence = float(conf)
                    cls_name = "person" if int(classes) == 0 else None
                    color = colors(int(classes))
                    custom_annotator.box_label(bbox_coords, label=None, color=tuple(reversed(color)))
                    custom_annotator.kpts(keypoints.data[0])

                    #
                    detection_data = create_detection_data(bbox, cls_name, confidence)
                    detection_results.append(detection_data)

                    # 거리 계산 (키포인트 사용)
                    distance = None
                    if keypoints is not None and len(keypoints.xy[0]) > 0:
                        kp = keypoints.xy[0][0]  # 첫 번째 키포인트 사용
                        x, y = map(int, kp[:2])
                        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                            distance = depth_map[y, x] * 9  # 스케일링 팩터 적용

                    text_lines = [
                        f"{cls_name if cls_name else 'Unknown'} {confidence * 100:.1f}%",  # 첫 줄: 객체 이름 + 신뢰도
                        f"Dist: {distance:.2f} m" if distance else "Dist: N/A"  # 둘째 줄: 거리 정보
                    ]

                    font_scale = 0.5
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_color = (255, 255, 255)
                    bg_color = (0, 0, 0)
                    line_spacing = 6

                    # 각 텍스트 크기 계산
                    text_sizes = [cv2.getTextSize(line, font, font_scale, 1)[0] for line in text_lines]
                    max_width = max(w for w, h in text_sizes)
                    total_height = sum(h for w, h in text_sizes) + (len(text_lines) - 1) * line_spacing

                    # 텍스트 위치 계산 (박스 내부 또는 외부)
                    text_x = xmin
                    text_y = ymin - total_height - 5 if ymin - total_height - 5 > 0 else ymax + 5

                    # 배경 사각형
                    cv2.rectangle(
                        img,
                        (text_x, text_y - 20),
                        (text_x + max_width + 10, text_y + total_height),
                        bg_color,
                        thickness=cv2.FILLED
                    )

                    # 텍스트 그리기
                    for i, line in enumerate(text_lines):
                        line_y = text_y + i * (text_sizes[i][1] + line_spacing)
                        cv2.putText(img, line, (text_x + 5, line_y), font, font_scale, font_color, 1)

        #image = cv2.addWeighted(overlay_frame, 0.5, img, 0.5, 0)


        # PIP로 깊이 맵 오버레이
        # C:\Users\08shw\anaconda3\Lib\site-packages\ultralytics\utils 에서 box_label에서
        # cv2.putText에서 label부분과 label의 배경 부분 cv2.rectangle 을 주석 처리
        # cv2.putText에서 label은 None으로 처리

        annotated_frame = custom_annotator.result()
        depth_map_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        #frame = picture_in_picture(img, depth_map_colored)
        frame = picture_in_picture(annotated_frame, depth_map_colored)

        end_time = time.time()
        fps_display = f"FPS: {10 / (end_time - start_time):.2f}"
        start_time = time.time()

        cv2.putText(frame, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('Pose Detection with Depth', frame)


        # 탐지 결과 JSON 출력 (stdout)
        if detection_results:
            print(json.dumps(detection_results), flush=True)
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = os.path.join('detection_result', f"detected_frame_{current_time}.jpg")
            cv2.imwrite(file_name, frame)
            #exit_flag = True
            sys.stdout.flush()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True  # 종료 플래그 설정

finally:
    # 비디오 캡처 해제
    if cap.isOpened():
        cap.release()

    # 비디오 파일 저장 해제
    out.release()

    # OpenCV 창 닫기
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)
