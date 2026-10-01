"""一次性腳本：從 Hugging Face 抓一小批 py150 真實 Python 檔案，落地成
code_snippet_import.py 的 `extract` 模式吃得下的「一個資料夾、很多 .py 檔」格式。
跑完這份資料就可以刪掉這個腳本（不是專案常駐程式碼）。

用法：
    pip install datasets
    python fetch_py150_sample.py --n 500 --out ../data/py150_sample
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="抽樣幾個檔案（不是全部 150k）")
    parser.add_argument("--out", type=Path, default=Path("../data/py150_sample"))
    args = parser.parse_args()

    from datasets import load_dataset
    ds = load_dataset("AISE-TUDelft/PY150k", split="train")

    print("資料集欄位：", ds.column_names)
    print("第一筆範例（前 300 字）：", str(ds[0])[:300])

    # 常見欄位名稱猜測，哪個存在就用哪個 —— 不同版本的 HF 鏡像欄位名可能不同
    text_col = next((c for c in ("code", "content", "text", "source") if c in ds.column_names), None)
    if text_col is None:
        raise SystemExit(f"找不到原始碼欄位，請看上面印出的欄位名稱,手動改這支腳本的 text_col")

    args.out.mkdir(parents=True, exist_ok=True)
    n = min(args.n, len(ds))
    written = 0
    for i in range(n):
        code = ds[i][text_col]
        if not isinstance(code, str) or not code.strip():
            continue
        (args.out / f"py150_{i:06d}.py").write_text(code, encoding="utf-8", errors="ignore")
        written += 1

    print(f"寫出 {written} 個 .py 檔案到 {args.out}")


if __name__ == "__main__":
    main()
