"""
訓練手勢辨識 MLP 模型
資料來源：collect_data.py 錄製的 data.csv
使用類別：one（滑動）、ok（點擊）、stop（回主畫面）、no_gesture
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import GESTURE_ACTION_MAP, MODEL_PATH, LABEL_PATH

DATA_PATH = "data.csv"

os.makedirs("model", exist_ok=True)

TARGET_CLASSES = list(GESTURE_ACTION_MAP.keys())  # ["one","ok","stop","no_gesture"]

# ── 載入 CSV ──────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"找不到 {DATA_PATH}，請先執行 collect_data.py 錄製資料"
    )

df = pd.read_csv(DATA_PATH)
df = df[df["label"].isin(TARGET_CLASSES)]
print(f"載入資料：{len(df)} 筆")
print("各類別數量：")
for cls, cnt in df["label"].value_counts().items():
    print(f"  {cls}: {cnt} 筆")

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

X = np.array(X, dtype=np.float32)
y = np.array(y)

# 類別分佈
for cls in TARGET_CLASSES:
    print(f"  {cls}: {(y == cls).sum()} 筆")

# ── Label Encoding ──────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"\n類別順序：{list(le.classes_)}")

# ── Train / Test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"訓練集：{len(X_train)}　測試集：{len(X_test)}\n")

# ── MLP 架構：63 → 128 → 64 → num_classes ─────────────────────
model = keras.Sequential([
    keras.layers.Input(shape=(63,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(num_classes, activation="softmax"),
], name="gesture_mlp")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── 訓練 ────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

# ── 評估 ────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n測試集準確率：{acc * 100:.2f}%")

# ── 儲存 ────────────────────────────────────────────────────────
model.save(MODEL_PATH)
with open(LABEL_PATH, "wb") as f:
    pickle.dump(le, f)

print(f"\n模型已儲存：{MODEL_PATH}")
print(f"標籤已儲存：{LABEL_PATH}")
print("\n接下來執行 main.py 開始手勢控制！")
