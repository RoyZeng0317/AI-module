"""train_grpo.py — GRPO 強化學習訓練迴圈，套用在專案自建的 GPT 模型上。

不用 trl.GRPOTrainer：trl 的 GRPOTrainer 綁定 HuggingFace transformers 的
PreTrainedModel/tokenizer 介面（AutoModelForCausalLM、.generate()、.config...），
而 tranning/transformer_chat.py 的 GPT 是從零手刻的 nn.Module，沒有這層介面，
把 model= 換成別的字串接不上；用外部 Qwen2.5-7B-Instruct 當基礎模型也違反
CLAUDE.md 規則06（禁用其他 AI 模型/API，全部自建）且遠超 RTX 4060 8G 算力。
這裡照 GRPO 論文（Shao et al., DeepSeekMath, 2024）的核心概念自己刻一份簡化版：
同一個 prompt 用目前的策略模型取樣出一組（group）completions，用「組內相對
排名」當 advantage（reward 減掉組內平均、除以組內標準差），不需要另外訓練一個
value/critic 模型；另外保留一份訓練開始前的凍結副本（reference model）算 KL
懲罰，避免策略在只有極少訓練資料時被少數幾筆 reward 訊號帶到跑掉。

Usage:
    python train_grpo.py --data sft_agent_data.jsonl --steps 50
"""

import argparse
import copy
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizer, EOS, SEP
from transformer_chat import GPT, DEFAULT_FINETUNE_DIR

DEFAULT_GRPO_DIR = Path(__file__).resolve().parent / "gpt_grpo_runs"


# --- 01. reward functions：驗證輸出是否為合法 JSON 工具呼叫、最終答案是否正確 ---

def json_format_reward(completion: str) -> float:
    """完整照搬原本的設計：有 ```json ...``` 區塊且能解析成 JSON 就加分。"""
    match = re.search(r"```json\s*(\{.*?\})\s*```", completion, re.DOTALL)
    if not match:
        return -1.0    # 未遵循 Tool 格式重扣分
    try:
        json.loads(match.group(1))
        return 1.0     # 格式正確給 + 1 分
    except json.JSONDecodeError:
        return -0.5    # JSON 解析失敗扣分


def accuracy_reward(completion: str, target_answer: str) -> float:
    """驗證最終答案是否包含正確數值。"""
    return 2.0 if str(target_answer) in completion else 0.0


def total_reward(completion: str, target_answer: str) -> float:
    return json_format_reward(completion) + accuracy_reward(completion, target_answer)


# --- 02. 訓練資料：sft_agent_data.jsonl 每行一筆 {"messages": [...]} ------------

def load_prompts(data_path: Path) -> list[dict]:
    """從每一行的 messages 取出 user 的題目與 tool 回傳的正確數值，組成
    (prompt, target_answer) 讓 GRPO 拿去取樣、評分——訓練時不需要 assistant
    示範回覆本身（那是 SFT 用的），GRPO 是讓模型自己生成、再用 reward 評分。
    """
    examples = []
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        messages = json.loads(line)["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        tool_msg = next((m["content"] for m in messages if m["role"] == "tool"), None)
        if tool_msg is None:
            continue
        examples.append({"prompt": user_msg, "target_answer": tool_msg})
    return examples


# --- 03. 載入策略模型（沿用 transformer_chat.py finetune() 產出的 checkpoint）---

def _load_policy(base_dir: Path, device: str):
    base_dir = Path(base_dir)
    model_path, config_path = base_dir / "model.pt", base_dir / "config.json"
    if not (model_path.exists() and config_path.exists() and (base_dir / "bpe_vocab.json").exists()):
        return None
    tokenizer = BPETokenizer.load(base_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = GPT(tokenizer.vocab_size, config["block_size"], config["n_layer"],
                config["n_embd"], config["n_head"], config.get("dropout", 0.0))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model, tokenizer, config


# --- 04. rollout：取樣一組 completions -----------------------------------------

@torch.no_grad()
def _sample_completion(model: GPT, prompt_ids: list[int], block_size: int,
                        max_new_tokens: int, temperature: float, device: str) -> list[int]:
    """跟 GPT.generate() 一樣是溫度取樣，但刻意不套 top-k/top-p 截斷：GRPO 之後
    要用「真實模型分佈」算這段 completion 的 log-prob 來做 policy gradient，取樣
    分佈跟算 log-prob 用的分佈必須是同一個，套了 top-k/top-p 兩者就對不上了。
    """
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == EOS:
            break
    return idx[0, len(prompt_ids):].tolist()


def _sequence_logprobs(model: GPT, prompt_ids: list[int], completion_ids: list[int],
                        block_size: int, device: str) -> torch.Tensor:
    """回傳這段 completion 裡每個 token 在 model 底下的 log-prob（teacher forcing
    forward，一個 token 一個 log-prob，之後才乘上 group advantage）。"""
    full = (prompt_ids + completion_ids)[-(block_size + 1):]
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x)
    log_probs = F.log_softmax(logits[0], dim=-1)
    n_completion = min(len(completion_ids), x.size(1) - (len(prompt_ids) - 1))
    n_completion = max(n_completion, 0)
    start = x.size(1) - n_completion
    targets = torch.tensor(full[1:], dtype=torch.long, device=device)[start:]
    token_log_probs = log_probs[start:].gather(1, targets.unsqueeze(1)).squeeze(1)
    return token_log_probs


# --- 05. 主訓練迴圈 ------------------------------------------------------------

def grpo_train(data_path: Path, base_dir: Path = DEFAULT_FINETUNE_DIR,
               out_dir: Path = DEFAULT_GRPO_DIR, steps: int = 50, group_size: int = 4,
               max_new_tokens: int = 64, temperature: float = 0.8, lr: float = 1e-5,
               beta: float = 0.04, save_every: int = 10, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    loaded = _load_policy(base_dir, device)
    if loaded is None:
        print(f"尚未找到可用的基礎 checkpoint：{base_dir}\n"
              "請先執行 `python transformer_chat.py pretrain --corpus ...` "
              "再執行 `finetune`，GRPO 是接著在 SFT checkpoint 上做強化學習微調。")
        return None

    model, tokenizer, config = loaded
    block_size = config["block_size"]
    ref_model = copy.deepcopy(model).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    examples = load_prompts(data_path)
    if not examples:
        print(f"{data_path} 沒有可用的訓練資料（需要至少一筆含 user/tool 訊息的 messages）。")
        return None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    out_dir = Path(out_dir)
    history: list[dict] = []

    for step in range(1, steps + 1):
        example = examples[(step - 1) % len(examples)]
        prompt_ids = (tokenizer.encode(example["prompt"]) + [SEP])[-block_size:]

        model.eval()
        completions = [_sample_completion(model, prompt_ids, block_size, max_new_tokens,
                                           temperature, device) for _ in range(group_size)]
        texts = [tokenizer.decode(c) for c in completions]
        rewards = torch.tensor([total_reward(t, example["target_answer"]) for t in texts],
                                dtype=torch.float32)

        mean_r, std_r = rewards.mean(), rewards.std()
        loss_val = None

        if std_r < 1e-6:
            # 整組 reward 都一樣（例如全部都答對或全部都答錯），組內相對排名沒有
            # 訊號可學，跳過這一步而不是除以幾乎是 0 的標準差製造出爆炸的 advantage。
            print(f"[grpo] step {step:4d}  mean_reward={mean_r.item():.3f}  (組內無差異，跳過更新)")
        else:
            advantages = (rewards - mean_r) / (std_r + 1e-6)

            model.train()
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            total_tokens = 0
            for completion_ids, advantage in zip(completions, advantages):
                if not completion_ids:
                    continue
                log_probs = _sequence_logprobs(model, prompt_ids, completion_ids, block_size, device)
                with torch.no_grad():
                    ref_log_probs = _sequence_logprobs(ref_model, prompt_ids, completion_ids, block_size, device)
                # GRPO 的 unbiased KL 估計式（Schulman, "Approximating KL Divergence"）：
                # k3 = exp(ref - cur) - (ref - cur) - 1，恆 >= 0，比 (ref-cur) 本身更穩定。
                log_ratio = ref_log_probs - log_probs
                kl = torch.exp(log_ratio) - log_ratio - 1
                token_loss = -(advantage.to(device) * log_probs - beta * kl)
                total_loss = total_loss + token_loss.sum()
                total_tokens += token_loss.numel()

            if total_tokens > 0:
                loss = total_loss / total_tokens
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_val = loss.item()
                print(f"[grpo] step {step:4d}  mean_reward={mean_r.item():.3f}  loss={loss_val:.4f}")

        history.append({"step": step, "mean_reward": mean_r.item(), "loss": loss_val})

        if step % save_every == 0 or step == steps:
            out_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save(out_dir)
            (out_dir / "config.json").write_text(json.dumps(config, indent=2))
            torch.save(model.state_dict(), out_dir / "model.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    return model, tokenizer, history


# --- CLI ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="自建 GPT 模型的 GRPO 強化學習訓練迴圈")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "sft_agent_data.jsonl")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_FINETUNE_DIR,
                         help="做為起點的 SFT checkpoint 目錄（transformer_chat.py finetune 的輸出）")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_GRPO_DIR)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04, help="KL 懲罰係數")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    grpo_train(args.data, args.base_dir, args.out_dir, args.steps, args.group_size,
               args.max_new_tokens, args.temperature, args.lr, args.beta, args.save_every)


if __name__ == "__main__":
    main()
