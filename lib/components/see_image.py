"""`/see <圖片路徑>`：讓 sinco 前端「看」一張圖——YOLO 偵測後用 image_analysis.format_report() 的
固定格式回報（確定性輸出，不交給 17M 模型生成，避免編造內容，見 ErrorLog #30/#33）。
自訂分類只讀 tranning/pcb_classifier_runs/（PCB 缺陷，pcb_dataset_prep.py 準備資料）；舊的
image_classifier_runs/ 是合成色塊測試 checkpoint（to_do #35），拿來分類真實照片會誤導，所以不讀。
沒有 PCB checkpoint 時只回報偵測結果。
讀取為純讀，可指向專案外路徑（同 /open、@ 附加，規則 01 只禁止「修改」）。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
PCB_DIR = _ROOT / "tranning" / "pcb_classifier_runs"  # 舊的合成色塊 image_classifier_runs/ 刻意不讀
NO_CLS = "（PCB 自訂分類器尚未訓練，見 tranning/pcb_dataset_prep.py）"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def see_image(arg: str) -> str:
    path = Path(arg.strip().strip('"')).expanduser()
    if not arg.strip():
        return "用法：/see <圖片路徑>"
    if not path.is_file():
        return f"找不到檔案：{path}"
    if path.suffix.lower() not in IMAGE_EXTS:
        return f"不是支援的圖片格式（{'、'.join(sorted(IMAGE_EXTS))}）：{path.name}"
    for d in (_ROOT / "tranning", _ROOT / "web" / "backend"):
        if str(d) not in sys.path:
            sys.path.append(str(d))
    try:
        import cv2
        from detector import detect
        from image_analysis import format_report
        from image_classifier_bnn import classify_image
        image = cv2.imread(str(path))
        if image is None:
            return f"無法讀取圖片：{path}"
        label, conf, ent = classify_image(path, out_dir=PCB_DIR) if (PCB_DIR / "best_model.pt").exists() else (NO_CLS, 0.0, 0.0)
        return f"[{path.name}]\n" + format_report({"detections": detect(image, conf=0.35),
                                                 "classification": {"label": label, "confidence": conf, "entropy": ent}})
    except Exception as exc:  # 缺套件/權重下載失敗等，回報而不是讓 UI 崩潰
        return f"圖片分析失敗：{type(exc).__name__}: {exc}"
