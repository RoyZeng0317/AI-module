"""語音辨識 CTC 對齊修復 pilot（獨立 runner；不改 speech_to_text.py / OCR.py 的 API 與正式權重）。

修的是 speech_to_text.train() 的四個對齊問題：
  1. 音訊被裁到 max_frames、文字卻完整保留 → 這裡用完整音訊，batch 內動態補零；
     超過 --max-audio-seconds 的整筆排除並記錄，不裁音訊。
  2. input_lengths 一律等於補零後長度 → 這裡用每筆真實的 CNN 輸出長度 (T//2)//2。
  3. BiLSTM 會讀到補零 → CNN 逐筆只跑有效段，再用 packed sequence 進 LSTM，
     每筆輸出與單獨跑該筆完全相同（測試會比對）。
  4. 未知字被靜默刪掉、zero_infinity 藏住無法對齊的樣本 →
     train 有舊字表沒有的字就整輪停止（不自動重建字表）；dev 的未知字不進 loss 但計入 CER；
     CTC 最短長度（文字長 + 相鄰重複字數）放不下就整筆排除並記錄；loss 出現 NaN/Inf 立即停止。
另外：BatchNorm 固定在 eval 模式（batch 4 的統計不可靠），訓練不做資料增強，
這兩點都寫進 status.json 供審查。輸出只寫到新的 run 資料夾，正式 speech_runs 不覆寫、不切換。
用法：
  python tranning/speech_ctc_pilot.py --train-manifest <train.json> --val-manifest <val.json>
      [--audio-root <dir>] [--regression-manifest <reg.json>]
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

TRANNING = Path(__file__).resolve().parent
ROOT = TRANNING.parent  # 專案根目錄
if str(TRANNING) not in sys.path:
    sys.path.insert(0, str(TRANNING))

import speech_to_text as stt  # noqa: E402
from OCR import BLANK, CRNN  # noqa: E402

DEFAULT_RUN_DIR = TRANNING / "speech_pilot_runs" / "alignment_v1"
MIN_SPEC_FRAMES = 4  # CRNN 的 CNN 在時間軸做兩次 2 倍池化，少於 4 幀會直接出錯


class PilotError(RuntimeError):
    """資料、權重或路徑不合格：拒絕執行。"""


class StopTraining(RuntimeError):
    """NaN/Inf/OOM：立即停止，不重跑。"""


# ---------------- 路徑與檔案 ----------------
def safe_path(raw, must_exist=False):
    p = Path(raw)
    resolved = p.resolve()
    root = Path(ROOT).resolve()
    if not resolved.is_relative_to(root):
        raise PilotError(f"path must stay inside the project folder: {raw}")
    parts = [x.lower() for x in p.parts] + [x.lower() for x in resolved.relative_to(root).parts]
    if any(x.startswith(".env") for x in parts):
        raise PilotError(f"path must not be or be inside a .env path: {raw}")
    if must_exist and not resolved.exists():
        raise PilotError(f"path not found: {raw}")
    return resolved


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise PilotError(f"manifest must be a non-empty JSON list: {path}")
    for i, r in enumerate(data):
        if not (isinstance(r, dict) and isinstance(r.get("audio"), str) and isinstance(r.get("text"), str)):
            raise PilotError(f"{path}[{i}] must have string 'audio' and 'text'")
    return data


# ---------------- CTC 長度規則 ----------------
def cnn_out_len(frames):
    return (frames // 2) // 2  # 已對 OCR.CRNN 實測 T=4..299 全部一致


def ctc_min_steps(ids):
    """CTC 最短需要的時間步數：每個字一步，相鄰重複字之間還要插一個 blank。"""
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def global_cer(pairs):
    """總編輯距離 ÷ 總參考字數（不是逐句 CER 平均）。"""
    chars = sum(len(ref) for ref, _ in pairs)
    return sum(edit_distance(hyp, ref) for ref, hyp in pairs) / chars if chars else None


# ---------------- 模型前向（逐筆 CNN + packed LSTM） ----------------
def forward_packed(model, specs, frames):
    """specs: list[(freq, T_i)]；回傳 (T_max', N, C) log_probs 與每筆真實輸出長度。
    CNN 逐筆只跑有效段，避免補零經由卷積與 BatchNorm 影響結果；LSTM 用 packed sequence。"""
    feats = []
    for spec, t in zip(specs, frames):
        conv = model.cnn(spec[:, :t].unsqueeze(0).unsqueeze(0))  # (1, 512, 1, T')
        feats.append(conv.squeeze(0).squeeze(1).transpose(0, 1))  # (T', 512)
    out_lens = torch.tensor([f.size(0) for f in feats])
    expected = torch.tensor([cnn_out_len(t) for t in frames])
    if not torch.equal(out_lens, expected):
        raise PilotError("CNN output length does not match (T//2)//2; CRNN changed, stopping")
    padded = pad_sequence(feats, batch_first=True)
    packed = pack_padded_sequence(padded, out_lens, batch_first=True, enforce_sorted=False)
    rnn_out, _ = model.rnn(packed)
    rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=padded.size(1))
    log_probs = torch.log_softmax(model.fc(rnn_out), dim=2)
    return log_probs.permute(1, 0, 2), out_lens


def greedy_decode(log_probs, out_lens, idx_to_char):
    """只解碼每筆有效長度內的輸出；原始結果，不做 remove_fillers 或任何補字。"""
    best = log_probs.argmax(2).transpose(0, 1).tolist()
    texts = []
    for seq, n in zip(best, out_lens.tolist()):
        chars, prev = [], None
        for idx in seq[:n]:
            if idx != BLANK and idx != prev:
                chars.append(idx_to_char.get(idx, ""))
            prev = idx
        texts.append("".join(chars))
    return texts


def set_train_mode(model):
    model.train()
    for m in model.modules():  # BatchNorm 固定使用舊權重的統計值
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


# ---------------- 資料準備 ----------------
def prepare(records, audio_root, charset, max_seconds, split, unknown_policy):
    """回傳 (items, excluded)。unknown_policy: 'reject'（train）或 'score_only'（dev/reg）。"""
    items, excluded, unknown = [], [], {}
    for r in records:
        path = safe_path(Path(audio_root) / r["audio"], must_exist=True)
        wave = stt.load_waveform(path)
        seconds = len(wave) / stt.SAMPLE_RATE
        if seconds > max_seconds:
            excluded.append({"audio": r["audio"], "reason": f"longer than {max_seconds} s", "seconds": seconds})
            continue
        spec = stt.waveform_to_spectrogram(wave)
        frames = spec.size(1)
        missing = sorted({c for c in r["text"] if c not in charset})
        for c in missing:
            unknown[c] = unknown.get(c, 0) + 1
        if missing and unknown_policy == "reject":
            continue  # 之後整輪停止
        ids = None if missing else [charset[c] for c in r["text"]]
        if frames < MIN_SPEC_FRAMES or (ids is not None and ctc_min_steps(ids) > cnn_out_len(frames)):
            excluded.append({"audio": r["audio"], "reason": "too short for CTC alignment",
                             "ctc_steps": cnn_out_len(frames), "needed": ctc_min_steps(ids or [])})
            continue
        if ids is not None and not ids:
            excluded.append({"audio": r["audio"], "reason": "empty transcript"})
            continue
        items.append({"audio": r["audio"], "text": r["text"], "ids": ids, "spec": spec,
                      "frames": frames, "seconds": round(seconds, 3)})
    if unknown and unknown_policy == "reject":
        raise PilotError(f"{split} has characters missing from the base charset: "
                         f"{''.join(sorted(unknown))[:80]}. Not rebuilding the charset; stopping.")
    return items, excluded, unknown


def batches(items, size, shuffle, gen=None):
    order = list(range(len(items)))
    if shuffle:
        gen.shuffle(order)
    return [[items[i] for i in order[k:k + size]] for k in range(0, len(order), size)]


def ctc_sum(model, batch, device, ctc):
    """回傳 (loss 總和, 參考字數, log_probs, out_lens)；只算有 ids 的樣本。"""
    specs = [it["spec"].to(device) for it in batch]
    log_probs, out_lens = forward_packed(model, specs, [it["frames"] for it in batch])
    keep = [k for k, it in enumerate(batch) if it["ids"] is not None]
    if not keep:
        return None, 0, log_probs, out_lens
    targets = torch.tensor([i for k in keep for i in batch[k]["ids"]], dtype=torch.long, device=device)
    tlens = torch.tensor([len(batch[k]["ids"]) for k in keep], dtype=torch.long)
    loss = ctc(log_probs[:, keep], targets, out_lens[keep], tlens)
    return loss, int(tlens.sum()), log_probs, out_lens


@torch.no_grad()
def evaluate(model, items, device, ctc, idx_to_char, batch_size):
    """train 與 dev 用同一套規則：CTC loss 總和 ÷ 參考字數；CER = 總編輯距離 ÷ 總字數。"""
    model.eval()
    loss_sum, chars, pairs, outputs = 0.0, 0, [], []
    for batch in batches(items, batch_size, shuffle=False):
        loss, n, log_probs, out_lens = ctc_sum(model, batch, device, ctc)
        if loss is not None:
            if not torch.isfinite(loss):
                raise StopTraining("non-finite loss during evaluation")
            loss_sum, chars = loss_sum + loss.item(), chars + n
        for it, hyp in zip(batch, greedy_decode(log_probs.cpu(), out_lens, idx_to_char)):
            pairs.append((it["text"], hyp))
            outputs.append({"audio": it["audio"], "reference": it["text"], "output": hyp, "seconds": it["seconds"]})
    return (round(loss_sum / chars, 6) if chars else None), global_cer(pairs), outputs


# ---------------- 主流程 ----------------
def run(args):
    run_dir = safe_path(args.run_dir)
    base_dir = safe_path(args.base_dir, must_exist=True)
    if run_dir == base_dir or run_dir.is_relative_to(base_dir):
        raise PilotError("run dir must not be inside the production speech_runs folder")
    if run_dir.exists():
        raise PilotError(f"run dir already exists; refusing to overwrite: {run_dir}")
    for f in ("charset.json", "config.json", "best_model.pt"):
        if not (base_dir / f).is_file():
            raise PilotError(f"base checkpoint is missing {f}; not training from scratch, stopping")
    base_sha = {f: sha256(base_dir / f) for f in ("charset.json", "config.json", "best_model.pt")}
    run_dir.mkdir(parents=True)
    status = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "decisions": {"batchnorm": "frozen in eval mode", "augmentation": "none",
                            "decode": "greedy, raw output, no remove_fillers",
                            "speaker_split": "no speaker_id in manifests; existing split only, not unseen-speaker"}}
    write_json(run_dir / "status.json", status)
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    code = 0
    try:
        status.update(body(args, run_dir, base_dir, log))
        status["status"] = "completed"
    except StopTraining as exc:
        status.update(status="stopped", reason=str(exc)); code = 3
    except PilotError as exc:
        status.update(status="rejected", reason=str(exc)); code = 2
    except Exception as exc:  # 其他錯誤（含讀檔失敗）也要留下紀錄，不重跑
        if "out of memory" not in str(exc).lower():
            status.update(status="failed", reason=f"{type(exc).__name__}: {exc}"); code = 4
        else:
            status.update(status="stopped", reason=f"out of memory: {exc}"); code = 3
    finally:
        after = {f: sha256(base_dir / f) for f in base_sha}
        status["base_checkpoint_sha256"] = after
        if after != base_sha:
            status.update(status="failed", reason="base checkpoint files changed during the run"); code = 4
        if torch.cuda.is_available():
            status["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(run_dir / "status.json", status)
        log_file.close()
    print(json.dumps({"ok": code == 0, "status": status["status"], "reason": status.get("reason"),
                      "run_dir": str(run_dir)}, ensure_ascii=True))
    return code


def body(args, run_dir, base_dir, log):
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)  # 與專案其他訓練相同的顯存上限
        torch.cuda.reset_peak_memory_stats()
    charset = json.loads((base_dir / "charset.json").read_text(encoding="utf-8"))
    config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    idx_to_char = {i: c for c, i in charset.items()}
    model = CRNN(len(charset) + 1, stt.FREQ_BINS, config["hidden_size"], args.dropout)
    try:
        model.load_state_dict(torch.load(base_dir / "best_model.pt", map_location="cpu"), strict=True)
    except RuntimeError as exc:
        raise PilotError(f"base checkpoint does not match charset/config: {exc}")
    model.to(device)

    train_path = safe_path(args.train_manifest, must_exist=True)
    val_path = safe_path(args.val_manifest, must_exist=True)
    train_root = safe_path(args.audio_root or train_path.parent, must_exist=True)
    val_root = safe_path(args.audio_root or val_path.parent, must_exist=True)
    reg_records = load_manifest(safe_path(args.regression_manifest, must_exist=True)) if args.regression_manifest else []
    reg_audio = {r["audio"] for r in reg_records}
    # 回歸片段（公開題）不可被抽進 train/dev
    train_pool = [r for r in load_manifest(train_path) if r["audio"] not in reg_audio]
    val_pool = [r for r in load_manifest(val_path) if r["audio"] not in reg_audio]
    rng = random.Random(args.seed)
    if len(train_pool) < args.train_count or len(val_pool) < args.dev_count:
        raise PilotError("manifest has fewer records than --train-count / --dev-count")
    train_rec = rng.sample(train_pool, args.train_count)
    dev_rec = rng.sample(val_pool, args.dev_count)

    train_items, train_ex, _ = prepare(train_rec, train_root, charset, args.max_audio_seconds, "train", "reject")
    dev_items, dev_ex, dev_unknown = prepare(dev_rec, val_root, charset, args.max_audio_seconds, "dev", "score_only")
    reg_root = safe_path(args.audio_root or Path(args.regression_manifest).resolve().parent) if reg_records else None
    reg_items, reg_ex, _ = prepare(reg_records, reg_root, charset, args.max_audio_seconds, "regression",
                                   "score_only") if reg_records else ([], [], {})
    if not train_items or not dev_items:
        raise PilotError("no usable train or dev items after checks")
    # val_loss 只算 ids 不為 None 的樣本（含未知字的句子無法當 CTC 目標）；val_cer 算全部 dev 樣本
    scored = [it for it in dev_items if it["ids"] is not None]
    population = {
        "val_loss": {"items": len(scored), "reference_chars": sum(len(it["text"]) for it in scored),
                     "rule": "CTC loss sum / reference chars, items whose text is fully in the charset"},
        "val_cer": {"items": len(dev_items), "reference_chars": sum(len(it["text"]) for it in dev_items),
                    "rule": "total edit distance / total reference chars, all dev items incl. unknown chars"},
        "train_loss": {"items": len(train_items), "reference_chars": sum(len(it["ids"]) for it in train_items)},
    }
    write_json(run_dir / "data_manifest.json", {
        "seed": args.seed, "train": [it["audio"] for it in train_items], "dev": [it["audio"] for it in dev_items],
        "regression": [it["audio"] for it in reg_items], "excluded": {"train": train_ex, "dev": dev_ex,
                                                                     "regression": reg_ex},
        "dev_unknown_chars": dev_unknown, "manifest_sha256": {"train": sha256(train_path), "val": sha256(val_path)},
        "metric_population": population,
        "note": "no test set; regression clips are public and not counted as unseen"})
    log(f"device={device} train={len(train_items)} dev={len(dev_items)} reg={len(reg_items)} "
        f"excluded train={len(train_ex)} dev={len(dev_ex)}")

    ctc = nn.CTCLoss(blank=BLANK, reduction="sum", zero_infinity=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    compare = dev_items[:args.compare_count] + reg_items

    _, _, baseline = evaluate(model, compare, device, ctc, idx_to_char, args.batch_size)
    tl, _, _ = evaluate(model, train_items, device, ctc, idx_to_char, args.batch_size)
    vl, vc, _ = evaluate(model, dev_items, device, ctc, idx_to_char, args.batch_size)
    history = [{"epoch": 0, "train_loss": tl, "val_loss": vl, "val_cer": vc, "updates": 0, "seconds": 0.0}]
    log(f"epoch 0 (base weights) {history[0]}")
    write_json(run_dir / "history.json", history)  # 每輪立即寫入，中途停止也留得下紀錄

    gen = random.Random(args.seed)
    start, updates, step_times, stop, epoch = time.monotonic(), 0, [], None, 0
    while stop is None:
        epoch += 1
        for batch in batches(train_items, args.batch_size, shuffle=True, gen=gen):
            if updates >= args.max_updates or time.monotonic() - start >= args.max_seconds:
                break  # 只在 batch 邊界檢查上限
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.monotonic()
            set_train_mode(model)
            optimizer.zero_grad(set_to_none=True)
            loss, n, _, _ = ctc_sum(model, batch, device, ctc)
            loss = loss / n  # 每個參考字的平均 loss，與報表同一個規則
            if not torch.isfinite(loss):
                raise StopTraining(f"non-finite loss at update {updates + 1}")
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not torch.isfinite(gnorm):
                raise StopTraining(f"non-finite gradient at update {updates + 1}")
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise StopTraining(f"non-finite parameter after update {updates + 1}")
            if device == "cuda":
                torch.cuda.synchronize()
            updates += 1
            step_times.append(time.monotonic() - t0)
            if len(step_times) == 6:
                per = sum(step_times[1:6]) / 5
                left = min(args.max_updates - updates, max(0.0, args.max_seconds - (time.monotonic() - start)) / per)
                log(f"benchmark: {per:.3f} s/update (updates 2-6); ETA about {left * per:.0f} s plus evaluation")
        tl, _, _ = evaluate(model, train_items, device, ctc, idx_to_char, args.batch_size)
        vl, vc, _ = evaluate(model, dev_items, device, ctc, idx_to_char, args.batch_size)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_cer": vc, "updates": updates,
                        "seconds": round(time.monotonic() - start, 2)})
        log(f"epoch {epoch} {history[-1]}")
        write_json(run_dir / "history.json", history)
        if updates >= args.max_updates:
            stop = "max_updates"
        elif time.monotonic() - start >= args.max_seconds:
            stop = "max_seconds"
        elif epoch >= args.epochs:
            stop = "max_epochs"

    torch.save(model.state_dict(), run_dir / "pilot_model.pt")
    (run_dir / "charset.json").write_text((base_dir / "charset.json").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(run_dir / "config.json", dict(config, pilot_note="pilot run; not a production checkpoint"))
    _, _, final = evaluate(model, compare, device, ctc, idx_to_char, args.batch_size)
    rows = [{"set": "regression" if b["audio"] in reg_audio else "dev", "audio": b["audio"],
             "reference": b["reference"], "seconds": b["seconds"],
             "baseline_output": b["output"], "final_output": f["output"]} for b, f in zip(baseline, final)]
    write_json(run_dir / "raw_compare.json", {"rows": rows, "note": "raw greedy output; regression is public, not unseen"})
    log(f"done: stop_reason={stop} updates={updates}")
    return {"stop_reason": stop, "updates": updates, "epochs": epoch, "metric_population": population, "train_seconds": round(time.monotonic() - start, 2),
            "steady_seconds_per_update": round(sum(step_times[1:]) / max(1, len(step_times) - 1), 4) if step_times else None,
            "config": {k: v for k, v in vars(args).items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="CTC alignment pilot: fine-tune a copy of the existing speech CRNN. "
                                             "Never writes to the production speech_runs folder.")
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--audio-root", default=None, help="Base folder for audio paths (default: each manifest's folder)")
    ap.add_argument("--regression-manifest", default=None, help="Public regression clips; reported only, not unseen")
    ap.add_argument("--base-dir", default=str(stt.DEFAULT_OUT_DIR), help="Existing checkpoint (read only)")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--train-count", type=int, default=256)
    ap.add_argument("--dev-count", type=int, default=64)
    ap.add_argument("--compare-count", type=int, default=8)
    ap.add_argument("--max-audio-seconds", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-updates", type=int, default=640)
    ap.add_argument("--max-seconds", type=float, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    for k in ("train_count", "dev_count", "batch_size", "epochs", "max_updates"):
        if getattr(args, k) < 1:
            ap.error(f"--{k.replace('_', '-')} must be at least 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        return run(args)
    except PilotError as exc:  # run dir 建立前的拒絕
        print(json.dumps({"ok": False, "status": "rejected", "reason": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
