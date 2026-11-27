# AI Human Image Editor（CLI）

本專案是可在 Windows 本機執行的影像處理 CLI 起始專案。它提供：

- 圖片資訊檢查
- 保留原始尺寸的降噪／輕度銳化
- 以遮罩指定區域進行物件修復（OpenCV Inpainting）
- 可選擇的前景／主體透明背景擷取（rembg）
- 合成測試與單元測試

**安全與功能界線：**本專案不提供移除人物衣物、推斷衣物下身體、或生成裸露身體的功能。人物相關處理以保留衣著的影像增強、背景分離及一般非敏感區域修復為限。

## 1. 系統需求

- Windows 10/11 64-bit
- 建議 Python 3.11（亦可嘗試 3.12）
- RTX 4060：本專案核心 OpenCV 工作流程不需要 GPU；可選 `rembg` 初始設定使用 CPU，避免 CUDA/PyTorch 版本衝突。
- 建議 8 GB 以上 RAM；高解析度圖片會使用較多記憶體。

> 「原始解析度」代表輸出不會主動縮小長寬；JPEG 重新編碼仍可能造成有損壓縮。建議輸出 PNG 以避免額外 JPEG 損失。

## 2. 安裝（PowerShell）

```powershell
# 解壓縮後進入專案資料夾
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows.ps1

# 啟用虛擬環境
.\.venv\Scripts\Activate.ps1

# 如需主體／前景擷取（首次會下載模型）
pip install -e ".[subject]"
```

若 `py -3.11` 不存在，請安裝 Python 3.11 x64，再重新執行安裝腳本。

## 3. CLI 使用

```powershell
aihe --help
aihe inspect "input.jpg"
aihe enhance "input.jpg" --output "outputs/enhanced.png"
aihe inpaint "input.jpg" --mask "mask.png" --output "outputs/repaired.png" --radius 5
aihe cutout "input.jpg" --output "outputs/subject.png"
```

### 遮罩規則

`mask.png` 必須與原圖長寬相同：
- 白色（或灰階大於 127）：要修復的區域
- 黑色：保留區域

遮罩可用任何繪圖軟體製作。請只對一般物件／背景瑕疵使用修復功能；不要用來移除人物衣物或生成衣物下的身體。

## 4. 指令說明

### inspect
讀取格式、尺寸、色彩模式與檔案大小。

### enhance
以 OpenCV 做輕度彩色降噪及可選銳化。預設不縮放圖片。
```powershell
aihe enhance input.jpg --output outputs/clean.png --strength 5 --sharpen 0.25
```
`strength` 建議 3–8；越高越可能抹除細節。`sharpen` 建議 0–0.5。

### inpaint
根據遮罩周圍像素填補一般物件或背景瑕疵。它不是生成式人體編輯模型，無法可靠地重建複雜內容。
```powershell
aihe inpaint input.jpg --mask mask.png --output outputs/fixed.png --radius 5
```

### cutout（選配）
安裝 `.[subject]` 後使用 rembg 擷取前景並輸出透明 PNG：
```powershell
aihe cutout input.jpg --output outputs/cutout.png
```
主體擷取模型可能把頭髮、毛髮或半透明物件判斷錯誤，需人工檢查。

## 5. 測試

```powershell
pip install -e ".[dev]"
pytest -q
```

## 6. 成功率與品質驗證

本專案不宣稱固定 90% 成功率。不同圖片、光線、遮罩品質與修復區域會造成很大差異。建議用至少 100 張合法且具代表性的測試圖片建立測試集，記錄：
- 任務完成率（輸出檔是否成功）
- 原始尺寸保持率
- 非遮罩區域像素差異
- 人工盲評品質分數

本專案的 `enhance` 與 `inpaint` 是傳統影像處理，不是可理解語意的通用生成式模型。

## 7. 常見錯誤

- `ModuleNotFoundError`：確認已啟用 `.venv`，再執行 `pip install -e .`
- `mask size must match image size`：遮罩長寬必須與原圖一致。
- `Could not decode image`：確認輸入是有效圖片，路徑加上引號。
- `rembg is not installed`：執行 `pip install -e ".[subject]"`
- 記憶體不足：先關閉其他大型程式；不要同時處理多張高解析度圖片。

## 8. 專案結構

```text
AI_Human_Image_Editor/
├── pyproject.toml
├── requirements.txt
├── requirements-subject.txt
├── install_windows.ps1
├── README.md
├── src/
│   └── ai_human_image_editor/
│       ├── __init__.py
│       ├── cli.py
│       └── imaging.py
└── tests/
    └── test_imaging.py
```
