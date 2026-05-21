"""
手勢資料蒐集工具（MediaPipe Tasks API）
操作方式：
  1 / 2 / 3 / 4 / 5  切換手勢類別
  空白鍵          開始 / 暫停錄製
  Q               離開並儲存
"""

import csv
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── 設定 ──────────────────────────────────────────────────────────
SAVE_PATH    = "data.csv"
SAMPLES_GOAL = 300
RECORD_DELAY = 0.05

CLASSES = {
    "1": "one",
    "2": "two_up",
    "3": "ok",
    "4": "stop",
    "5": "no_gesture",
}
CLASS_COLORS = {
    "one":        (0,   200, 255),
    "two_up":     (180,   0, 255),
    "ok":         (0,   255, 120),
    "stop":       (255, 180,   0),
    "no_gesture": (160, 160, 160),
}

# 手部連線（21 個關節點的連接關係，用來畫骨架）
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ── 下載模型（只需一次）────────────────────────────────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("下載 hand_landmarker 模型（約 8 MB）...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("模型下載完成\n")


# ── 手部骨架繪製 ──────────────────────────────────────────────────
def draw_hand(frame, landmarks, h, w):
    for s, e in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)


# ── 正規化 ────────────────────────────────────────────────────────
def normalize_landmarks(landmarks):
    lm = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    lm -= lm[0]
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-8
    return (lm / scale).flatten().tolist()


# ── 初始化 ────────────────────────────────────────────────────────
download_model()

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ── 載入已有資料 ──────────────────────────────────────────────────
counts      = {cls: 0 for cls in CLASSES.values()}
file_exists = os.path.exists(SAVE_PATH)

if file_exists:
    with open(SAVE_PATH, "r") as f:
        for row in csv.DictReader(f):
            if row["label"] in counts:
                counts[row["label"]] += 1
    print(f"載入已有資料：{counts}")

csvfile = open(SAVE_PATH, "a", newline="")
writer  = csv.writer(csvfile)
if not file_exists:
    header = [f"{ax}{i}" for i in range(21) for ax in ("x","y","z")] + ["label"]
    writer.writerow(header)

# ── 主迴圈 ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("無法開啟攝影機")

current_class = "one"
recording     = False
last_record   = 0.0
start_time    = time.time()

print("\n按 1-4 切換手勢，空白鍵開始/暫停，Q 離開")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result       = landmarker.detect_for_video(mp_image, timestamp_ms)

    detected = bool(result.hand_landmarks)

    if detected:
        lms = result.hand_landmarks[0]
        draw_hand(frame, lms, h, w)

        now = time.time()
        if recording and (now - last_record) >= RECORD_DELAY:
            if counts[current_class] < SAMPLES_GOAL:
                writer.writerow(normalize_landmarks(lms) + [current_class])
                csvfile.flush()
                counts[current_class] += 1
                last_record = now
            else:
                recording = False

    # ── HUD ─────────────────────────────────────────────────────
    color = CLASS_COLORS[current_class]
    cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 20), -1)

    status_str = "REC ●" if recording else "PAUSE"
    status_col = (0, 0, 220) if recording else (120, 120, 120)
    cv2.putText(frame, status_str,      (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Gesture: {current_class}", (110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,      2, cv2.LINE_AA)

    hand_col = (0, 255, 100) if detected else (80, 80, 80)
    cv2.putText(frame, "Hand: OK" if detected else "Hand: --",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{counts[current_class]}/{SAMPLES_GOAL}",
                (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    ratio = min(counts[current_class] / SAMPLES_GOAL, 1.0)
    cv2.rectangle(frame, (0, 75), (int(w * ratio), 82), color, -1)
    cv2.rectangle(frame, (0, 75), (w, 82), (60, 60, 60), 1)

    y_off = h - 10
    for cls in reversed(list(CLASSES.values())):
        done  = counts[cls] >= SAMPLES_GOAL
        c_col = (0, 220, 80) if done else CLASS_COLORS[cls]
        mark  = " OK" if done else f" {counts[cls]}"
        cv2.putText(frame, f"{cls}{mark}", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_col, 1, cv2.LINE_AA)
        y_off -= 22

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        recording = not recording
        print(f"{'開始' if recording else '暫停'} 錄製：{current_class}")
    elif chr(key) in CLASSES:
        current_class = CLASSES[chr(key)]
        recording     = False
        print(f"切換至：{current_class}（已有 {counts[current_class]} 筆）")

cap.release()
cv2.destroyAllWindows()
landmarker.close()
csvfile.close()

print("\n最終統計：")
for cls, cnt in counts.items():
    bar = "█" * (cnt * 20 // SAMPLES_GOAL)
    print(f"  {cls:<15} {cnt:>3}/{SAMPLES_GOAL}  {bar}")
print(f"\n資料儲存至：{SAVE_PATH}")
