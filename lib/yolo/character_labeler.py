"""獨立人物標註／訓練／偵測入口：python -m lib.yolo.character_labeler。"""
import tkinter as tk
from tkinter import ttk

from lib.yolo.object_labeler import Labeler
from object_labeling import LabelDataset
from character_weights import DATASET, RUNS, WEIGHTS, publish_weights


class CharacterLabeler(Labeler):
    def __init__(self, root):
        dataset = LabelDataset(DATASET)
        if not dataset.index.exists():
            dataset.data["classes"] = ["人物1"]
            dataset.persist()
        self.runs_root = RUNS
        super().__init__(root, DATASET)
        root.title("人物特徵標註與獨立 characters.pt 模型")
        self.status.set("新增姓名或代號後框選人物／人臉；請保持同一標註方式。尚非可靠身份驗證。")
        if WEIGHTS.is_file() and WEIGHTS.stat().st_size > 0:
            self.best.set(str(WEIGHTS))
        self.checkpoint_choice = tk.StringVar(value="last")
        options = ttk.Frame(root); options.pack(fill="x")
        ttk.Label(options, text="人物發布權重（依驗證結果選擇）").pack(side="left")
        ttk.Combobox(options, textvariable=self.checkpoint_choice, values=("best", "last"), state="readonly", width=6).pack(side="left")

    def training_finished(self, best):
        source = best.with_name(self.checkpoint_choice.get() + ".pt")
        path = publish_weights(source)
        from detector import _model
        _model.cache_clear()  # 同一路徑發布新人物權重後，下次偵測不可沿用舊模型快取
        self.best.set(str(path))
        self.status.set(f"人物權重已輸出：{path}。舊檔已備份；可開始即時偵測，仍需檢查誤認。")


def main():
    root = tk.Tk()
    CharacterLabeler(root)
    root.mainloop()


if __name__ == "__main__":
    main()
