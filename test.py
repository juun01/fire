import torch
import cv2
import os
import time
import re
import threading
from ultralytics import YOLO

# 모델 로드
model = YOLO("C:/Users/PC/Downloads/best (1).pt")
model.fuse()

save_directory = "C:/fire_capture"
os.makedirs(save_directory, exist_ok=True)

image_count_lock = threading.Lock()
frame_lock = threading.Lock()
result_lock = threading.Lock()

shared_frame = None
shared_result = None
image_count = 0
last_save_time = 0

# 프레임 저장 시작 인덱스 계산
def get_start_index(directory):
    max_index = -1
    pattern = re.compile(r"fire_(\d+)\.jpg")
    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx
    return max_index + 1

image_count = get_start_index(save_directory)

# 화면 영역 계산용 (초기 1프레임으로 크기 계산)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라 열기 실패")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("프레임 캡처 실패")
H, W, _ = frame.shape
half_W = W // 2
zones = {
    "A": (0, 0, half_W, H),
    "B": (half_W, 0, W, H)
}
cap.release()

# 타이머 초기화
last_print = {("fire", "A"): 0, ("fire", "B"): 0, ("light", "A"): 0, ("light", "B"): 0}
fire_start_time = {"A": None, "B": None}
last_display_time = 0

# 겹침 확인 함수
def is_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

# 프레임 캡처 스레드
def capture_thread():
    global shared_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                shared_frame = frame.copy()

# 추론 스레드
def inference_thread():
    global shared_frame, shared_result
    while True:
        with frame_lock:
            frame = shared_frame.copy() if shared_frame is not None else None
        if frame is None:
            continue
        result = model(frame, imgsz=416, conf=0.2, verbose=False)
        with result_lock:
            shared_result = (frame, result)

# 디스플레이 및 로직 스레드
def display_thread():
    global shared_result, image_count, last_save_time, last_display_time
    while True:
        time.sleep(0.01)
        now = time.time()

        with result_lock:
            if shared_result is None:
                continue
            frame, results = shared_result

        detected_fire_zones = set()

        for name, (x1, y1, x2, y2) in zones.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{name} Area ", (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label not in ("fire", "light"):
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_box = (x1, y1, x2, y2)

                for zone_name, roi_box in zones.items():
                    if not is_overlap(det_box, roi_box):
                        continue

                    key = (label, zone_name)
                    if label == "fire":
                        detected_fire_zones.add(zone_name)
                        if fire_start_time[zone_name] is None:
                            fire_start_time[zone_name] = now
                        duration = now - fire_start_time[zone_name]
                        if duration >= 3.0 and now - last_print[key] > 1.0:
                            print(f"{zone_name}구역에 불이 났습니다 ({int(duration)}초 이상 감지됨)")
                            last_print[key] = now

                        if duration >= 3.0 and now - last_save_time > 10.0:
                            file_name = os.path.join(save_directory, f"fire_{image_count}.jpg")
                            cv2.imwrite(file_name, frame)
                            print(f"저장 완료: {file_name}")
                            image_count += 1
                            last_save_time = now

                    elif label == "light" and now - last_print[key] > 1.0:
                        print(f"{zone_name}구역에서 빛이 감지되었습니다")
                        last_print[key] = now

                    text_pos = (x1, max(y1 - 10, 20))
                    color = (0, 0, 255) if label == "fire" else (255, 255, 0)
                    cv2.putText(frame, f"{zone_name}:{label.upper()}", text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

        for zone in zones:
            if zone not in detected_fire_zones:
                fire_start_time[zone] = None

        if now - last_display_time >= 0.2:
            cv2.imshow("Fire & Light Detection", frame)
            last_display_time = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# 스레드 실행
threading.Thread(target=capture_thread, daemon=True).start()
threading.Thread(target=inference_thread, daemon=True).start()
threading.Thread(target=display_thread, daemon=True).start()

# 메인 스레드는 대기만
while True:
    time.sleep(1)
