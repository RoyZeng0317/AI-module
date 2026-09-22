"""自訂 YOLO 檢測系統骨架：微調專屬類別 -> 檢測 -> 規則判斷 -> 中文報告（可接 sinco / GUI）。

目前沒有真實資料集，所以先用合成資料（圓/方/三角）驗證「產資料 -> 微調 -> 檢測 -> 規則」整條管線能跑，
不代表真實準確度（同 circuit_diagram_train.py 的「骨架先行、資料後補」）。有真實資料時：準備 YOLO 格式
data.yaml（images/ + labels/ 各 train/val），`--train data.yaml` 即可；規則 JSON 改成自己的類別與數量。
權重沿用 web/backend/detector.py 的 detect()（支援換權重），不新增第二套推論。

Usage:
    python yolo_custom.py --make-data tranning/yolo_data --n 200
    python yolo_custom.py --train tranning/yolo_data/data.yaml --epochs 30
    python yolo_custom.py --inspect photo.jpg --rules '{"circle": 2, "square": 1}'
"""
import argparse, json, random, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "web" / "backend"), str(Path(__file__).resolve().parent)]

import cv2, numpy as np  # noqa: E402
from bayesian_utils import low_confidence_warning  # noqa: E402

CLASSES = ["circle", "square", "triangle"]
RUNS = Path(__file__).resolve().parent / "yolo_runs"
BEST = RUNS / "custom" / "weights" / "best.pt"
REVIEW_BELOW = 0.5  # 低於此信心度的偵測不計入規則，標成「待人工確認」


def make_synthetic(out: Path, n: int = 200, size: int = 320, seed: int = 0) -> Path:
    """隨機背景上畫 1~4 個幾何形狀，同步寫 YOLO 標註（class cx cy w h，0~1 正規化）。"""
    rng, out = random.Random(seed), Path(out)
    for split, cnt in (("train", n), ("val", max(n // 5, 4))):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(cnt):
            img, lines = np.full((size, size, 3), rng.randint(20, 90), np.uint8), []
            for _ in range(rng.randint(1, 4)):
                c, r, (x, y) = rng.randrange(3), rng.randint(18, 40), (rng.randint(45, size - 45), rng.randint(45, size - 45))
                col = tuple(rng.randint(140, 255) for _ in range(3))
                if c == 0: cv2.circle(img, (x, y), r, col, -1)
                elif c == 1: cv2.rectangle(img, (x - r, y - r), (x + r, y + r), col, -1)
                else: cv2.fillPoly(img, [np.array([(x, y - r), (x - r, y + r), (x + r, y + r)])], col)
                lines.append(f"{c} {x / size:.5f} {y / size:.5f} {2 * r / size:.5f} {2 * r / size:.5f}")
            cv2.imwrite(str(out / "images" / split / f"{i}.jpg"), img)
            (out / "labels" / split / f"{i}.txt").write_text("\n".join(lines))
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" +
                    "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return yaml


def train(data_yaml: Path = None, epochs: int = 30, imgsz: int = 320, base: str = "yolo26n.pt", resume: bool = False) -> Path:
    """以預訓練 yolo26n 為起點微調專屬類別（RTX 4060 8G 綽綽有餘）；回傳 best.pt。
    resume=True 從 last.pt（Ultralytics 每個 epoch 都會存）接續，資料集/epochs 沿用上次的設定。"""
    from ultralytics import YOLO
    from detector import _device
    if resume:
        YOLO(str(BEST.with_name("last.pt"))).train(resume=True)
        return BEST
    YOLO(base).train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=_device(), workers=0,
                     project=str(RUNS), name="custom", exist_ok=True, verbose=False)
    return BEST


def detach_train(argv: list[str]) -> int:
    """把同一條訓練指令丟到獨立子程序（關掉 VSCode／終端機也不會中斷），輸出寫 yolo_runs/train.log。"""
    import subprocess
    RUNS.mkdir(parents=True, exist_ok=True)
    flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    with open(RUNS / "train.log", "ab") as log:
        return subprocess.Popen([sys.executable, __file__, *[a for a in argv if a != "--detach"]], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, creationflags=flags, cwd=str(_ROOT)).pid


def status() -> str:
    """讀 results.csv + args.yaml 回報目前 epoch / 總 epoch / 最新 mAP50。"""
    import csv, yaml
    csv_path, args_path = RUNS / "custom" / "results.csv", RUNS / "custom" / "args.yaml"
    if not csv_path.exists():
        return "尚無訓練紀錄（還沒開始，或第一個 epoch 還沒跑完）。log：" + str(RUNS / "train.log")
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(csv_path, encoding="utf-8"))]
    total = yaml.safe_load(open(args_path, encoding="utf-8")).get("epochs", "?") if args_path.exists() else "?"
    last = rows[-1]
    return f"epoch {last['epoch']}/{total}  mAP50={float(last['metrics/mAP50(B)']):.3f}  mAP50-95={float(last['metrics/mAP50-95(B)']):.3f}"


def parse_rules(s: str) -> dict[str, int]:
    """支援 JSON 或 circle=2,square=1（PowerShell 會吃掉 JSON 的雙引號，所以提供第二種寫法）。"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {k.strip(): int(v) for k, v in (kv.split("=") for kv in s.split(",") if kv.strip())}


def apply_rules(detections: list[dict], expect: dict[str, int]) -> dict:
    """規則判斷：各類別數量是否符合預期。低信心偵測不計數，另列待確認（避免亂判）。"""
    sure = [d for d in detections if d["confidence"] >= REVIEW_BELOW]
    unsure = [d for d in detections if d["confidence"] < REVIEW_BELOW]
    got = {k: sum(d["label"] == k for d in sure) for k in {*expect, *(d["label"] for d in sure)}}
    issues = [f"{k}：預期 {expect.get(k, 0)} 個，偵測到 {v} 個（{'缺少' if v < expect.get(k, 0) else '多出'} {abs(v - expect.get(k, 0))}）"
              for k, v in sorted(got.items()) if v != expect.get(k, 0)]
    return {"ok": not issues, "counts": got, "issues": issues, "unsure": unsure}


def inspect(image_path: Path, expect: dict[str, int], weights: str = None, conf: float = 0.25) -> str:
    """檢測一張圖 -> 規則判斷 -> 固定格式中文報告（確定性輸出，不交給小模型生成，避免編造）。"""
    from detector import detect
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"無法讀取圖片：{image_path}")
    res = apply_rules(detect(image, conf=conf, weights=weights or str(BEST)), expect)
    lines = ["判定：合格" if res["ok"] else "判定：不合格", "數量：" + "、".join(f"{k}×{v}" for k, v in sorted(res["counts"].items()))]
    lines += res["issues"] + [f'待人工確認：{d["label"]}（{d["confidence"]:.0%}）{low_confidence_warning(d["confidence"])}' for d in res["unsure"]]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="自訂 YOLO 檢測系統（合成資料 / 微調 / 規則檢測）")
    p.add_argument("--make-data", type=Path); p.add_argument("--n", type=int, default=200)
    p.add_argument("--train", type=Path); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--inspect", type=Path); p.add_argument("--rules", default="{}", help="預期數量，如 circle=2,square=1 或 JSON")
    p.add_argument("--detach", action="store_true", help="背景執行訓練（log: yolo_runs/train.log）")
    p.add_argument("--resume", action="store_true", help="從 last.pt 接續上次訓練")
    p.add_argument("--status", action="store_true", help="查看訓練進度")
    a = p.parse_args()
    if a.make_data: print("資料集：", make_synthetic(a.make_data, a.n))
    if (a.train or a.resume) and a.detach: print(f"已在背景啟動（PID {detach_train(sys.argv[1:])}），用 --status 查看進度，log：{RUNS / 'train.log'}")
    elif a.train or a.resume: print("權重：", train(a.train, a.epochs, resume=a.resume))
    if a.status: print(status())
    if a.inspect: print(inspect(a.inspect, parse_rules(a.rules)))


if __name__ == "__main__":
    main()
