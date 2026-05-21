# 手勢辨識電腦控制系統

利用攝影機即時辨識手勢，透過 MediaPipe + MLP 神經網路將手部動作映射為滑鼠操控，不需要任何硬體設備就能用手勢控制電腦。

---

## 支援手勢

| 手勢 | 動作 | 說明 |
|------|------|------|
| `one` — 食指朝上 | 移動滑鼠 | 食指尖位置對應全螢幕，EMA 平滑避免抖動 |
| `two_up` — 食指＋中指朝上 | 上下滾動 / 左右換頁 | 上下為虛擬卡點滾輪；左右水平移動觸發 Alt+←/→ |
| `ok` — OK 手勢 | 左鍵點擊 | 在滑鼠目前位置點擊 |
| `stop` — 張開手掌 | 顯示桌面 | 觸發 Win+D |

---

## 系統架構

```
攝影機影像
    │
    ▼
MediaPipe Hand Landmarker          ← 偵測 21 個手部關節點 (x, y, z)
    │
    ▼
正規化特徵（63 維）                ← 以手腕為原點，除以最大距離縮放
    │
    ▼
MLP 分類器                         ← 63 → 128 → 64 → 5 類別
    │
    ▼
GESTURE_ACTION_MAP                 ← 手勢 → 動作對應
    │
    ▼
PyAutoGUI 執行動作
```

---

## 檔案說明

```
.
├── collect_data.py    # 用攝影機蒐集手勢訓練資料
├── train.py           # 訓練 MLP 分類模型
├── main.py            # 即時手勢辨識 + 電腦控制（主程式）
├── config.py          # 所有參數設定（手勢對應、門檻值、滑鼠靈敏度等）
├── data.csv           # 訓練資料（由 collect_data.py 產生）
├── hand_landmarker.task  # MediaPipe 手部偵測模型（自動下載）
└── model/
    ├── gesture_mlp.keras  # 訓練好的 MLP 模型
    └── labels.pkl         # 類別標籤編碼器
```

---

## 安裝

Python 3.10+ 環境下執行：

```bash
pip install -r requirements.txt
```

**相依套件：**
- `mediapipe` — 手部關節點偵測
- `opencv-python` — 攝影機影像擷取與 HUD 顯示
- `tensorflow` — MLP 模型訓練與推論
- `scikit-learn` — 資料前處理與 Label Encoding
- `pyautogui` — 滑鼠、鍵盤控制
- `numpy` / `pandas` — 資料處理

---

## 使用流程

### Step 1 — 蒐集訓練資料

```bash
python collect_data.py
```

| 按鍵 | 功能 |
|------|------|
| `1` | 切換至 `one`（食指） |
| `2` | 切換至 `two_up`（雙指） |
| `3` | 切換至 `ok` |
| `4` | 切換至 `stop`（手掌） |
| `5` | 切換至 `no_gesture`（無手勢） |
| `空白鍵` | 開始 / 暫停錄製 |
| `Q` | 離開並儲存 |

每個類別目標蒐集 **300 筆**，螢幕上會顯示即時進度條。資料存至 `data.csv`。

### Step 2 — 訓練模型

```bash
python train.py
```

訓練完成後模型儲存至 `model/gesture_mlp.keras`，同時輸出測試集準確率。

### Step 3 — 啟動手勢控制

```bash
python main.py
```

啟動攝影機視窗，即時顯示偵測結果與觸發的動作。按 `Q` 離開。

---

## 參數調整

所有參數集中在 [`config.py`](config.py)，不需要改動程式碼：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `CONFIDENCE_THRESHOLD` | `0.85` | 信心值低於此值視為無手勢，調高減少誤觸 |
| `COOLDOWN_SEC` | `1.0` | 點擊 / 桌面動作的冷卻秒數 |
| `MOUSE_SMOOTH` | `0.4` | 滑鼠 EMA 平滑係數，越小越平滑但反應慢 |
| `MOUSE_MARGIN` | `0.15` | 畫面邊緣忽略區，避免滑鼠卡角落 |
| `NOTCH_SIZE` | `0.025` | 滾動卡點距離，調小讓滾動更靈敏 |
| `NOTCH_SCROLL` | `8` | 每個卡點的滾動格數 |
| `SWIPE_THRESHOLD` | `0.10` | 左右換頁的最小位移距離 |

---

## 模型架構

```
Input (63,)          ← 21 個關節點 × (x, y, z)
Dense(128, ReLU)
BatchNormalization
Dropout(0.3)
Dense(64, ReLU)
Dropout(0.2)
Dense(5, Softmax)    ← one / two_up / ok / stop / no_gesture
```

- 優化器：Adam
- 損失函數：Sparse Categorical Crossentropy
- Early Stopping（patience=5）+ ReduceLROnPlateau（patience=3）

---

## 注意事項

- 第一次執行時會自動下載 `hand_landmarker.task`（約 8 MB）
- 建議在光線充足的環境使用，手部偵測較穩定
- `pyautogui.FAILSAFE` 已停用，滑鼠不會因移到角落而拋出例外
