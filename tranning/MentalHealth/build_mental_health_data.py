"""把 data/pairs_emotion_draft.json 的 prompt 轉成 Overfitting.py 需要的
{"prompt", "score"} 格式。score 借用 action.py 已訓練好的 0~5 情緒迴歸模型
(emotion_runs/) 自動標記，不用手動逐筆標籤。
"""
import json, sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from action import predict_emotion  # noqa: E402

SRC = _PROJECT_ROOT / "data" / "pairs_emotion_draft.json"
DST = _MODULE_DIR / "mental_health_data.json"

pairs = json.loads(SRC.read_text(encoding="utf-8"))
out = []
for item in pairs:
    result = predict_emotion(item["prompt"])
    if result["label"] is None:
        raise RuntimeError(result["status"])
    out.append({"prompt": item["prompt"], "score": result["label"]})

DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"寫入 {len(out)} 筆到 {DST}")
