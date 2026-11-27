# -*- coding: utf-8 -*-
"""
範例程式：使用 Hugging Face Transformers 載入輕量開源模型進行文字生成

這個範例會：
1. 自動偵測目前環境是否有可用的 NVIDIA GPU（CUDA），若有則使用 GPU 加速，否則退回使用 CPU。
2. 從 Hugging Face Hub 下載並載入輕量模型 distilgpt2（檔案體積小，適合本機測試）。
3. 給定一段中文/英文提示詞，讓模型接續生成一段文字，並印出結果。

第一次執行時，程式會自動從網路下載模型檔案（約數百 MB），
下載完成後會被快取在使用者資料夾（預設路徑：C:\\Users\\<使用者名稱>\\.cache\\huggingface），
之後再次執行就不需要重新下載。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def main():
    # 步驟一：偵測裝置（GPU 優先，沒有 GPU 就用 CPU）
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"偵測到可用的 GPU：{gpu_name}，將使用 GPU 進行推論。")
    else:
        device = "cpu"
        print("未偵測到可用的 GPU，將使用 CPU 進行推論（速度會比較慢，屬正常現象）。")

    # 步驟二：載入模型與 tokenizer
    # distilgpt2 是 GPT-2 的輕量蒸餾版本，檔案小、下載快，適合用來驗證環境是否設置成功。
    model_name = "distilgpt2"
    print(f"正在載入模型：{model_name}（第一次執行會需要從網路下載，請耐心等候）...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # 步驟三：建立文字生成 pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,  # pipeline 的 device 參數：0 代表第一張 GPU，-1 代表 CPU
    )

    # 步驟四：給定提示詞，進行文字生成
    prompt = "Once upon a time, in a small village,"
    print(f"\n提示詞（prompt）：{prompt}")
    print("正在生成文字，請稍候...\n")

    outputs = generator(
        prompt,
        max_new_tokens=50,       # 最多再生成 50 個 token
        num_return_sequences=1,  # 只需要產生一組結果
        do_sample=True,          # 開啟取樣，讓生成結果更有變化
        temperature=0.8,         # 溫度參數，數值越高生成結果越有創意（也越不穩定）
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("=== 生成結果 ===")
    print(outputs[0]["generated_text"])
    print("================\n")
    print("範例程式執行完成！代表本機的 PyTorch + Transformers 開發環境已經可以正常運作。")


if __name__ == "__main__":
    main()
