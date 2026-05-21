"""
即時手勢辨識 + 電腦控制（MediaPipe Tasks API）
執行前請先跑 collect_data.py → train.py
按 Q 離開
"""

import os
import pickle
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    GESTURE_ACTION_MAP, MODEL_PATH, LABEL_PATH,
    CONFIDENCE_THRESHOLD, COOLDOWN_SEC,
    NOTCH_SIZE, NOTCH_SCROLL,
    SWIPE_FRAMES, SWIPE_THRESHOLD, COOLDOWN_LR_SEC,
    MOUSE_SMOOTH, MOUSE_MARGIN,
)

# ── 手部連線 ──────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

MODEL_TASK_PATH = "hand_landmarker.task"
MODEL_TASK_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ── 下載模型 ──────────────────────────────────────────────────────
def download_model():
    if os.path.exists(MODEL_TASK_PATH):
        return
    print("下載 hand_landmarker 模型（約 8 MB）...")
    urllib.request.urlretrieve(MODEL_TASK_URL, MODEL_TASK_PATH)
    print("模型下載完成\n")


# ── 手部骨架繪製 ──────────────────────────────────────────────────
def draw_hand(frame, landmarks, h, w):
    for s, e in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 255, 255), -1)


# ── 正規化 ────────────────────────────────────────────────────────
def normalize_landmarks(landmarks):
    lm = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    lm -= lm[0]
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-8
    return (lm / scale).flatten().reshape(1, -1)


# ── 動作後立刻重新置頂 ───────────────────────────────────────────
# ── 執行動作 ──────────────────────────────────────────────────────
def do_scroll(mid_y, anchor):
    """
    虛擬卡點滾輪：手指位移超過一個卡點距離才觸發一次滾動。
    anchor[0] 儲存目前的錨點 Y 值，手勢開始時設為 None 讓此函式自動初始化。
    回傳 (顯示文字, 顯示顏色)
    """
    if anchor[0] is None:
        anchor[0] = mid_y
        return None, None

    delta = anchor[0] - mid_y          # 正值 = 手往上 = 頁面往上
    notches = int(delta / NOTCH_SIZE)  # 累積了幾個卡點

    if notches == 0:
        return None, None

    pyautogui.scroll(notches * NOTCH_SCROLL)
    anchor[0] -= notches * NOTCH_SIZE  # 錨點前進，不重置（保留餘量）

    if notches > 0:
        return "Scroll Up", (255, 220, 0)
    else:
        return "Scroll Down", (255, 120, 0)


def do_lr(swipe_buf, last_lr):
    """左右換頁（離散，有冷卻），回傳 (觸發成功, 文字, 顏色, 更新後的 last_lr)"""
    if len(swipe_buf) < SWIPE_FRAMES:
        return False, None, None, last_lr
    if time.time() - last_lr < COOLDOWN_LR_SEC:
        return False, None, None, last_lr
    dx = swipe_buf[-1][0] - swipe_buf[0][0]
    dy = swipe_buf[-1][1] - swipe_buf[0][1]
    if abs(dx) > abs(dy) and abs(dx) > SWIPE_THRESHOLD:
        if dx < 0:
            pyautogui.hotkey("alt", "left")
            return True, "Swipe Left (Back)", (200, 255, 100), time.time()
        else:
            pyautogui.hotkey("alt", "right")
            return True, "Swipe Right (Fwd)", (100, 255, 200), time.time()
    return False, None, None, last_lr


def do_action(action, tip_buf):
    """處理 HOME / CLICK（SWIPE 由主迴圈單獨處理）"""
    if action == "HOME":
        pyautogui.hotkey("win", "d")
        return True, "Show Desktop", (0, 200, 255)
    if action == "CLICK":
        pyautogui.click()
        return True, "Click", (0, 255, 120)
    return False, None, None


# ── 初始化 ────────────────────────────────────────────────────────
download_model()

print("載入模型中...")
gesture_model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "rb") as f:
    le = pickle.load(f)
print(f"類別：{list(le.classes_)}\n")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_TASK_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)
pyautogui.FAILSAFE = False

# ── 主迴圈 ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("無法開啟攝影機")

cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
_TOPMOST = getattr(cv2, "WND_PROP_TOPMOST", 5)
cv2.setWindowProperty("Gesture Control", _TOPMOST, 1)

print("手勢控制啟動！按 Q 離開\n")
print("  one    (食指朝上)       → 移動滑鼠游標")
print("  two_up (食指＋中指朝上) → 上下左右滑動")
print("  ok     (OK 手勢)        → 左鍵點擊")
print("  stop   (張開手掌)       → 顯示桌面 (Win+D)\n")

last_trigger  = 0.0
last_lr       = 0.0
tip_buffer    = deque(maxlen=SWIPE_FRAMES)  # 左右換頁用
scroll_anchor = [None]                      # 卡點滾輪錨點，[None] 用 list 讓函式內可修改
status_text   = "Standby"
status_color  = (180, 180, 180)
start_time    = time.time()
screen_w, screen_h = pyautogui.size()
smooth_pos = [screen_w / 2, screen_h / 2]   # EMA 平滑後的滑鼠座標 [x, y]

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

    gesture = "no_gesture"
    conf    = 0.0

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        draw_hand(frame, lms, h, w)

        tip = lms[8]   # 食指尖
        tip_buffer.append((tip.x, tip.y))

        feat = normalize_landmarks(lms)
        pred = gesture_model.predict(feat, verbose=0)[0]
        idx  = int(np.argmax(pred))
        conf = float(pred[idx])
        gesture = le.classes_[idx] if conf >= CONFIDENCE_THRESHOLD else "no_gesture"

        # MOUSE：食指尖 EMA 平滑移動滑鼠
        if gesture == "one":
            scroll_anchor[0] = None
            norm_x = max(0.0, min(1.0, (lms[8].x - MOUSE_MARGIN) / (1 - 2 * MOUSE_MARGIN)))
            norm_y = max(0.0, min(1.0, (lms[8].y - MOUSE_MARGIN) / (1 - 2 * MOUSE_MARGIN)))
            smooth_pos[0] += MOUSE_SMOOTH * (norm_x * screen_w - smooth_pos[0])
            smooth_pos[1] += MOUSE_SMOOTH * (norm_y * screen_h - smooth_pos[1])
            pyautogui.moveTo(int(smooth_pos[0]), int(smooth_pos[1]))
            status_text  = "Mouse Move"
            status_color = (180, 0, 255)

        # SWIPE：two_up 中點追蹤，分上下（卡點滾輪）和左右（離散）
        elif gesture == "two_up":
            mid_x = (lms[8].x + lms[12].x) / 2
            mid_y = (lms[8].y + lms[12].y) / 2
            tip_buffer.append((mid_x, mid_y))

            # 上下：卡點滾輪
            text, color = do_scroll(mid_y, scroll_anchor)
            if text:
                status_text, status_color = text, color

            # 左右：離散換頁
            triggered, text, color, last_lr = do_lr(tip_buffer, last_lr)
            if triggered:
                status_text, status_color = text, color
                tip_buffer.clear()
        else:
            scroll_anchor[0] = None   # 手勢結束，重置錨點
    else:
        tip_buffer.clear()
        scroll_anchor[0] = None

    now      = time.time()
    can_fire = (now - last_trigger) >= COOLDOWN_SEC
    action   = GESTURE_ACTION_MAP.get(gesture, "NONE")

    if can_fire and action not in ("NONE", "MOUSE", "SWIPE"):
        triggered, text, color = do_action(action, tip_buffer)
        if triggered:
            status_text  = text
            status_color = color
            last_trigger = now

    # ── HUD ─────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    gesture_display = gesture if gesture != "no_gesture" else "—"
    cv2.putText(frame, f"Gesture: {gesture_display}  ({conf:.2f})",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Action : {status_text}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)

    if gesture == "one" and len(tip_buffer) < SWIPE_FRAMES:
        ratio = len(tip_buffer) / SWIPE_FRAMES
        cv2.rectangle(frame, (0, 70), (int(w * ratio), 76), (100, 200, 255), -1)

    cv2.imshow("Gesture Control", frame)
    cv2.setWindowProperty("Gesture Control", _TOPMOST, 1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("程式結束。")
