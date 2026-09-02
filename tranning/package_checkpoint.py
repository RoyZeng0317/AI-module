"""package_checkpoint.py — bundle a trained transformer_chat.py checkpoint
into a single portable .zip that can be copied to a different machine and
run there without needing the rest of this repo.

Why this exists: a checkpoint directory (model.pt/config.json/bpe_vocab.json
/bpe_merges.json) is not runnable by itself -- it needs transformer_chat.py
and bpe_tokenizer.py's *code* (the GPT class definition, BPETokenizer) to be
loaded back into a working model. This script copies exactly those two
source files alongside the checkpoint files into one zip, plus a short
README with the run commands, so the whole thing is self-contained.

Cross-hardware note (why this matters beyond "it's convenient"): the
checkpoint itself is just tensors -- `torch.save(model.state_dict())` has
no GPU vendor baked into it. transformer_chat.py's `_resolve_device()` picks
CUDA (NVIDIA) > XPU (Intel Arc, native `torch.xpu` backend) > CPU
automatically at load time, so the exact same package trained on this
machine's NVIDIA card will pick up an Intel Arc Pro card on a future machine
without any code change -- provided that machine's PyTorch build actually
has Intel GPU support installed (see the README this script writes).

Usage:
    python package_checkpoint.py --checkpoint-dir gpt_code_pretrain_runs --out dist/gpt_code_pretrain_runs.zip
"""

import argparse
import zipfile
from pathlib import Path

_TRANNING_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = _TRANNING_DIR / "dist"

# the checkpoint's own files -- whatever a pretrain()/finetune() run wrote.
CHECKPOINT_FILES = ("model.pt", "config.json", "bpe_vocab.json", "bpe_merges.json", "history.json")
# the source files needed to load the checkpoint back into a working model
# on a machine that doesn't have the rest of this repo.
SOURCE_FILES = ("transformer_chat.py", "bpe_tokenizer.py")

README_TEMPLATE = """checkpoint package: {name}

內容物：
  {checkpoint_files}          -- 訓練好的權重/tokenizer/設定檔
  transformer_chat.py, bpe_tokenizer.py -- 載入並執行這個 checkpoint 需要的程式碼

需求（目標機器上要先裝好）：
  pip install torch
  -- 想吃 Intel Arc Pro 的話，torch 要是有 XPU 支援的版本（PyTorch 官方
     從 2.4 開始原生支援 `torch.xpu`；沒有的話會自動退回 CPU 推論，不會
     報錯，只是比較慢）。程式碼會自動偵測：CUDA > XPU > CPU，不用手動指定。

跑法（把這個 zip 解壓縮，在解壓縮出來的資料夾裡執行）：
  # 這個 checkpoint 目前只做過 pretrain()（無監督下一個字元/token 預測），
  # 還沒有 finetune() 過，所以是「接續程式碼片段」而不是「一問一答」：
  python transformer_chat.py complete --prompt "def get_user(" --out-dir . --max-new-tokens 60

  # 如果之後這個 checkpoint 有另外跑過 finetune()，會多出一份 finetune 用的
  # checkpoint 資料夾，那個要用 chat 指令（一問一答）：
  # python transformer_chat.py chat --out-dir .
"""


def package_checkpoint(checkpoint_dir: Path, out_path: Path,
                        source_dir: Path = _TRANNING_DIR) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    out_path = Path(out_path)

    missing = [f for f in CHECKPOINT_FILES if f != "history.json" and not (checkpoint_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{checkpoint_dir} 缺少必要檔案 {missing}，看起來不是一個完整的 "
            f"pretrain()/finetune() checkpoint（訓練中途的資料夾不會被打包，"
            f"避免打包出一半新一半舊的壞掉狀態）。"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in CHECKPOINT_FILES:
            path = checkpoint_dir / name
            if path.exists():
                zf.write(path, arcname=name)
                included.append(name)
        for name in SOURCE_FILES:
            path = source_dir / name
            zf.write(path, arcname=name)
            included.append(name)

        readme = README_TEMPLATE.format(
            name=checkpoint_dir.name,
            checkpoint_files=", ".join(f for f in CHECKPOINT_FILES if f in included),
        )
        zf.writestr("README.txt", readme)
        included.append("README.txt")

    return {
        "out_path": str(out_path),
        "included_files": included,
        "size_bytes": out_path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bundle a transformer_chat.py checkpoint + loader code into a portable .zip"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                         help=f"預設寫到 {DEFAULT_OUT_DIR}/<checkpoint 資料夾名稱>.zip")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_path = args.out or (DEFAULT_OUT_DIR / f"{checkpoint_dir.name}.zip")

    summary = package_checkpoint(checkpoint_dir, out_path)
    size_mb = summary["size_bytes"] / (1024 * 1024)
    print(f"打包完成：{summary['out_path']}（{size_mb:.1f} MB）")
    print("內含檔案：")
    for name in summary["included_files"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
