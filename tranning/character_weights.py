"""人物模型獨立位置與可回復權重發布；不執行訓練。"""
import shutil, uuid
from pathlib import Path

from object_labeling import ROOT, project_path

DATASET = ROOT / "data" / "character_detection"
RUNS = ROOT / "tranning" / "character_detection_runs"
WEIGHTS = ROOT / "tranning" / "yolo_runs" / "custom" / "weights" / "characters.pt"


def publish_weights(source, target=WEIGHTS):
    source, target = project_path(source), project_path(target)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("人物權重尚未產生，空白.pt不能作為模型。")
    if source == target:
        raise ValueError("來源與輸出權重不可相同。")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = project_path(target.parent / "characters_backup" / uuid.uuid4().hex / target.name)
        backup.parent.mkdir(parents=True)
        shutil.copy2(target, backup)
    temporary = project_path(target.with_name(f".characters-{uuid.uuid4().hex}.tmp"))
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
