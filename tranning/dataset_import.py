"""dataset_import.py — turn plain files you already have (paired .txt files,
image/audio files with a matching .txt transcript, or video recordings) into
the JSON manifests chats.py / OCR.py / speech_to_text.py / voice_clone.py
expect, or into the "one subfolder per class" image layout data_split.py /
road_sign_train.py / image_classifier_bnn.py expect — without hand-writing
JSON first. This is the fix for CLAUDE.md 需求 #04: 手上沒有現成 JSON，但有
一般的音訊/影片/文字檔案。

Wired into train_gui.py's per-task tabs as a "步驟零：匯入原始檔案"
step, same idea as kicad_dataset_convert.py's "auto convert, no manual
labelling" step for circuit diagrams. Character trait scores still need a
human judgment call per entry (numeric sliders) — those stay GUI-only
(prompt_traits.py); this script only covers entries where the raw file
itself already fully determines the pair (the text/transcript IS the label).

Modes (pick exactly one with --mode):

  pairs          <source>/*.txt -> [{"prompt": ..., "reply": ...}, ...]
                 供 chats.py（含角色專屬聊天 checkpoint）使用。
                 每個 .txt 檔案的格式二選一（用哪一種由檔案內容自動判斷）：
                   (a) 第一行是 prompt，接一個空行，之後（可多行）是 reply
                   (b) 只有兩行：第一行 prompt，第二行 reply
                 檔名本身不重要，只有檔案內容有意義。

  sidecar-image  <source>/*.{png,jpg,jpeg,bmp} + 同檔名的 .txt
                 -> [{"image": 檔名, "text": ...}, ...] 供 OCR.py 使用。

  sidecar-audio  <source>/*.wav（必須是 16-bit PCM）+ 同檔名的 .txt
                 -> [{"audio": 檔名, "text": ...}, ...] 供
                 speech_to_text.py / voice_clone.py 使用（兩者 manifest
                 格式相同，一份匯入結果兩邊都能吃）。不是 16-bit PCM 的
                 .wav 會被跳過並列在警告清單裡，不會讓整個匯入失敗。

  video-frames   <source>/<類別名稱>/*.{mp4,avi,mov,mkv,webm}
                 -> <out>/<類別名稱>/*.jpg（每隔 --interval 秒擷取一張
                 影格）。不產生 manifest，直接產生 data_split.py 吃的
                 「一個子資料夾一個類別」圖片資料夾，接著就能餵給
                 road_sign_train.py 或 image_classifier_bnn.py。

  md-chat        <source>/*.md（Voyager 匯出的 Gemini/ChatGPT 對話紀錄，
                 格式為 "## Turn N" + "### 👤 使用者" / "### 🤖 助理"）
                 -> [{"prompt": ..., "reply": ...}, ...] 供 chats.py 使用。
                 同一個 Turn 沒有助理回覆的使用者訊息會先累積起來，直到
                 下一次出現助理回覆才合併成一組 prompt/reply；連續多筆
                 「不同問題、但助理回覆逐字相同」的組合（Voyager 匯出時
                 偶爾會把上一則回覆重複貼到後面幾個 Turn 上，是匯出工具
                 的 bug，不是真的重複回答）只保留第一次出現，避免模型
                 學到「不管問什麼都回同一段話」。

--val-ratio > 0（僅 pairs / sidecar-image / sidecar-audio / md-chat 適用）：
把蒐集到的 entries 隨機洗牌切成兩份，寫成 <out 去掉副檔名>_train.json /
_val.json，而不是像 CLAUDE.md to-do #14 那樣把 train 複製一份充當 val
（那樣量少於 2 筆才會退回單一檔案，並印出提示）。

Usage:
    python dataset_import.py --mode pairs --source <資料夾> --out <manifest.json>
    python dataset_import.py --mode pairs --source <資料夾> --out <manifest.json> --val-ratio 0.2
    python dataset_import.py --mode sidecar-image --source <資料夾> --out <manifest.json>
    python dataset_import.py --mode sidecar-audio --source <資料夾> --out <manifest.json> --val-ratio 0.2
    python dataset_import.py --mode video-frames --source <資料夾> --out <輸出資料夾> --interval 1.0
    python dataset_import.py --mode md-chat --source <資料夾> --out <manifest.json>
"""

import argparse
import json
import random
import re
import wave
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
AUDIO_EXTS = {".wav"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MARKDOWN_EXTS = {".md"}

_MD_TURN_RE = re.compile(r"^## Turn \d+\s*$", re.MULTILINE)
_MD_ROLE_HEADER_RE = re.compile(r"^### (?:👤 使用者|🤖 助理)\s*$", re.MULTILINE)
_MD_FOOTER_RE = re.compile(r"\n-{3,}\s*\n\*Exported from.*", re.DOTALL)
_MD_USER_HEADER = "### 👤 使用者"
_MD_ASSISTANT_HEADER = "### 🤖 助理"


def is_16bit_pcm_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getsampwidth() == 2
    except (wave.Error, OSError):
        return False


def split_prompt_reply(text: str) -> tuple[str, str] | None:
    """解析單一 .txt 檔案的內容成 (prompt, reply)，兩種格式都接受
    （見模組 docstring）；格式不符（少於兩個非空段落）回傳 None。
    """
    text = text.strip()
    if not text:
        return None
    if "\n\n" in text:
        prompt, _, reply = text.partition("\n\n")
    else:
        lines = text.splitlines()
        if len(lines) < 2:
            return None
        prompt, reply = lines[0], "\n".join(lines[1:])
    prompt, reply = prompt.strip(), reply.strip()
    if not prompt or not reply:
        return None
    return prompt, reply


def collect_pairs(source: Path) -> tuple[list[dict], list[str]]:
    """掃描 source 底下所有 *.txt，回傳 (entries, skipped_filenames)。"""
    entries = []
    skipped = []
    for path in sorted(source.glob("*.txt")):
        parsed = split_prompt_reply(path.read_text(encoding="utf-8"))
        if parsed is None:
            skipped.append(f"{path.name}（格式不符，需要至少 prompt+reply 兩段內容）")
            continue
        prompt, reply = parsed
        entries.append({"prompt": prompt, "reply": reply})
    return entries, skipped


def collect_sidecar(source: Path, kind: str) -> tuple[list[dict], list[str]]:
    """掃描 source 底下的圖片/音檔，各自找同檔名的 .txt 當作 text。
    kind 是 "image" 或 "audio"，同時也是輸出 dict 裡的欄位名稱
    （"image": 檔名 / "audio": 檔名）。回傳 (entries, skipped_filenames)。
    """
    exts = IMAGE_EXTS if kind == "image" else AUDIO_EXTS
    entries = []
    skipped = []
    for path in sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in exts):
        if kind == "audio" and not is_16bit_pcm_wav(path):
            skipped.append(f"{path.name}（不是 16-bit PCM WAV，speech_to_text.py/voice_clone.py 只吃這個格式）")
            continue
        sidecar = path.with_suffix(".txt")
        if not sidecar.exists():
            skipped.append(f"{path.name}（找不到同檔名的 .txt 文字稿）")
            continue
        text = sidecar.read_text(encoding="utf-8").strip()
        if not text:
            skipped.append(f"{path.name}（對應的 .txt 是空的）")
            continue
        entries.append({kind: path.name, "text": text})
    return entries, skipped


def _extract_role_text(segment: str, header: str) -> str | None:
    """從一個 Turn 區塊裡取出指定角色（使用者/助理）標題後面的內文，
    直到下一個角色標題（不是任意 "### " 子標題，例如助理回覆裡常見的
    "### 📚 相關詞彙" 這種內容小標題必須留在內文裡，不能被切斷）或區塊結尾
    為止。找不到該角色標題時回傳 None。
    """
    idx = segment.find(header)
    if idx == -1:
        return None
    line_end = segment.find("\n", idx)
    rest = segment[line_end + 1:] if line_end != -1 else ""
    next_header = _MD_ROLE_HEADER_RE.search(rest)
    if next_header:
        rest = rest[:next_header.start()]
    text = rest.strip()
    return text or None


def parse_md_chat(text: str) -> list[dict]:
    """把一份 Voyager 匯出的 Gemini/ChatGPT 對話 Markdown 轉成
    [{"prompt": ..., "reply": ...}, ...]。

    連續好幾個 Turn 都只有使用者訊息、沒有助理回覆時，先累積這些使用者
    訊息（用換行接起來），直到出現下一個助理回覆才配對成一組——這對應
    使用者連續補充問題、助理只回一次的真實對話情境。助理回覆逐字重複
    （匯出工具的已知 bug）只保留第一次出現。
    """
    text = _MD_FOOTER_RE.sub("", text)
    segments = _MD_TURN_RE.split(text)[1:]

    entries = []
    seen_replies = set()
    pending_user: list[str] = []
    for segment in segments:
        user_text = _extract_role_text(segment, _MD_USER_HEADER)
        assistant_text = _extract_role_text(segment, _MD_ASSISTANT_HEADER)
        if user_text:
            pending_user.append(user_text)
        if assistant_text:
            if pending_user and assistant_text not in seen_replies:
                entries.append({"prompt": "\n".join(pending_user), "reply": assistant_text})
                seen_replies.add(assistant_text)
            pending_user = []
    return entries


def collect_md_chat(source: Path) -> tuple[list[dict], list[str]]:
    """掃描 source 底下所有 *.md 對話匯出檔，回傳 (entries, skipped_filenames)。"""
    entries = []
    skipped = []
    for path in sorted(source.glob("*.md")):
        parsed = parse_md_chat(path.read_text(encoding="utf-8"))
        if not parsed:
            skipped.append(f"{path.name}（沒有偵測到任何使用者/助理成對內容）")
            continue
        entries.extend(parsed)
    return entries, skipped


def write_manifest(entries: list[dict], out: Path, val_ratio: float = 0.0, seed: int = 42) -> dict:
    """把 entries 寫成 manifest。若 out 已存在，會先讀出既有內容並合併去重
    （同一份原始檔案資料夾重複匯入時不會產生重複條目）。

    val_ratio<=0，或去重後總筆數不足 2 筆（沒辦法真的切出 held-out
    val），就只寫一個檔案到 out；否則洗牌後切成 <out 去掉副檔名>_train.json
    / _val.json 兩個檔案。回傳統計摘要（CLI 印出訊息、測試斷言都會用到）。
    """
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
        for e in existing:
            if e not in entries:
                entries.append(e)

    if val_ratio <= 0 or len(entries) < 2:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "total": len(entries), "train_path": str(out), "val_path": None,
            "train_count": len(entries), "val_count": 0,
        }

    shuffled = entries[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    n_val = min(n_val, len(shuffled) - 1)  # 至少留 1 筆給 train
    val_entries = shuffled[:n_val]
    train_entries = shuffled[n_val:]

    train_path = out.with_name(f"{out.stem}_train{out.suffix}")
    val_path = out.with_name(f"{out.stem}_val{out.suffix}")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(train_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "total": len(entries), "train_path": str(train_path), "val_path": str(val_path),
        "train_count": len(train_entries), "val_count": len(val_entries),
    }


def extract_video_frames(source: Path, out: Path, interval: float = 1.0, image_ext: str = ".jpg") -> dict:
    """source/<類別>/*.mp4(...) -> out/<類別>/*.jpg，每隔 interval 秒存一張
    影格。回傳 {類別名稱: 擷取張數}。只有這個 mode 需要 opencv，其餘 mode
    完全不會 import cv2。

    存檔刻意不用 cv2.imwrite(path, frame)：實測發現 Windows 上的 OpenCV
    對非 ASCII 路徑（這個專案的類別資料夾名稱本來就常常是中文，例如
    「停止」「讓路」）會靜默寫入失敗——回傳值雖然可以檢查，但目錄底下
    真的完全沒有檔案，不是丟例外，很容易誤以為擷取成功。改用
    cv2.imencode() 編碼成 bytes 後用 Path.write_bytes() 自己寫檔，繞過
    OpenCV 內部的檔案路徑處理，中文路徑一樣能正確寫入（已用真的中文
    類別名稱資料夾實測驗證）。
    """
    import cv2

    counts: dict[str, int] = {}
    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        videos = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not videos:
            continue
        target_dir = out / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_interval = max(1, round(fps * interval)) if fps > 0 else 1
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % frame_interval == 0:
                    encoded_ok, buf = cv2.imencode(image_ext, frame)
                    if encoded_ok:
                        frame_path = target_dir / f"{video_path.stem}_{frame_idx:06d}{image_ext}"
                        frame_path.write_bytes(buf.tobytes())
                        saved += 1
                frame_idx += 1
            cap.release()
        counts[class_dir.name] = saved
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True,
                         choices=["pairs", "sidecar-image", "sidecar-audio", "video-frames", "md-chat"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                         help="pairs/sidecar-* = 輸出 manifest JSON 路徑；video-frames = 輸出圖片資料夾")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                         help="pairs/sidecar-* 適用：>0 時把結果切成 train/val 兩個檔案")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=float, default=1.0, help="video-frames 適用：每隔幾秒擷取一張影格")
    args = parser.parse_args()

    if args.mode == "video-frames":
        counts = extract_video_frames(args.source, args.out, interval=args.interval)
        total = sum(counts.values())
        print(f"共擷取 {total} 張影格：")
        for name, n in counts.items():
            print(f"  {name}: {n} 張")
        if not counts:
            print(f"沒有在 {args.source} 底下找到任何「子資料夾 + 影片」，請確認每個類別各自一個子資料夾。")
        return

    if args.mode == "pairs":
        entries, skipped = collect_pairs(args.source)
    elif args.mode == "sidecar-image":
        entries, skipped = collect_sidecar(args.source, "image")
    elif args.mode == "sidecar-audio":
        entries, skipped = collect_sidecar(args.source, "audio")
    else:
        entries, skipped = collect_md_chat(args.source)

    if not entries:
        print(f"沒有在 {args.source} 底下找到任何可用的檔案。")

    summary = write_manifest(entries, args.out, val_ratio=args.val_ratio, seed=args.seed)
    print(f"共匯入 {summary['total']} 筆")
    if summary["val_path"]:
        print(f"  train: {summary['train_path']}（{summary['train_count']} 筆）")
        print(f"  val:   {summary['val_path']}（{summary['val_count']} 筆）")
    else:
        print(f"  寫入: {summary['train_path']}")
        if args.val_ratio > 0:
            print("  （筆數不足 2 筆，無法切出 held-out val，全部寫進同一個檔案）")
    if skipped:
        print(f"略過 {len(skipped)} 筆：")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
