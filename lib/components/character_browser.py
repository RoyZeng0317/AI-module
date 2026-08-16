"""character_browser.py — `/character` 指令開啟的角色卡瀏覽器（CLAUDE.md
需求 #02：「另外的 GUI 介面，不是與 home_screen.py 同一個 GUI」）。

列出 tranning/characters/*.json（character_model.py --build 產生的角色
卡），顯示描述 + 氣質分數；按「切換到這個角色」會呼叫傳入的
conversation.set_persona()，換掉 home_screen.py 主聊天視窗接下來要用的
persona checkpoint。這是一個獨立的 Toplevel 視窗（跟 home_screen.py 的
Claude 風格聊天版面完全不同的版面/配色），不是把角色清單塞進聊天視窗裡的
某個分頁——但共用同一個 conversation 物件（同一個 process），選好角色立刻
反映在主視窗，不需要另外跑一個程序、也不用做任何 IPC。

目前所有角色共用同一個 character_chat_runs checkpoint（見 CLAUDE.md
to-do #14 的已知限制、跟 app/components/cli.py 的 switch_character() 一致，
還沒做到「一個角色各自獨立聊天 checkpoint」）。
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from chats import DEFAULT_OUT_DIR

TRANNING_DIR = Path(__file__).resolve().parents[2] / "tranning"
CHARACTERS_DIR = TRANNING_DIR / "characters"
CHARACTER_CHAT_DIR = CHARACTERS_DIR / "character_chat_runs"
DEFAULT_PERSONA_NAME = "sinco"

BG = "#191a1b"
PANEL_BG = "#232425"
FG = "white"
MUTED_FG = "#9aa0a6"
ACCENT = "#3b82f6"
WARN = "#c98b3a"


def load_character_cards() -> list[dict]:
    if not CHARACTERS_DIR.exists():
        return []
    cards = []
    for path in sorted(CHARACTERS_DIR.glob("*.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if card.get("name"):
            cards.append(card)
    return cards


def _build_card_row(parent: tk.Widget, card: dict, on_switch) -> tk.Frame:
    row = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="solid")

    tk.Label(row, text=card["name"], bg=PANEL_BG, fg=FG, font=("Segoe UI", 11, "bold"),
              anchor="w").pack(fill="x", padx=10, pady=(8, 2))

    description = card.get("description", "")
    if description:
        tk.Label(row, text=description, bg=PANEL_BG, fg=MUTED_FG, anchor="w",
                  justify="left", wraplength=480).pack(fill="x", padx=10)

    traits: dict = card.get("traits", {})
    if traits:
        ranked = sorted(traits.items(), key=lambda kv: -kv[1])[:6]
        trait_text = "、".join(f"{name} {score:.0%}" for name, score in ranked)
        tk.Label(row, text=f"氣質：{trait_text}", bg=PANEL_BG, fg=ACCENT, anchor="w",
                  wraplength=480, justify="left").pack(fill="x", padx=10, pady=(2, 0))

    if not (CHARACTER_CHAT_DIR / "encoder.pt").exists():
        tk.Label(row, text="（尚未訓練這個角色的聊天 checkpoint，切換後會顯示「尚未訓練」提示而非真的回覆）",
                  bg=PANEL_BG, fg=WARN, anchor="w", wraplength=480, justify="left").pack(
            fill="x", padx=10, pady=(2, 0))

    ttk.Button(row, text="切換到這個角色", command=on_switch).pack(anchor="e", padx=10, pady=8)
    return row


def open_character_browser(parent: tk.Tk, conversation) -> tk.Toplevel:
    """開啟角色卡瀏覽器視窗，回傳建立好的 Toplevel（方便測試檢查內容/呼叫切換）。"""
    win = tk.Toplevel(parent)
    win.title("sinco 角色卡瀏覽器")
    win.geometry("560x480")
    win.configure(bg=BG)

    tk.Label(win, text="角色卡瀏覽器", bg=BG, fg=FG, font=("Segoe UI", 13, "bold"),
              anchor="w").pack(fill="x", padx=14, pady=(14, 2))
    current_var = tk.StringVar(value=f"目前人格：{conversation.persona}")
    tk.Label(win, textvariable=current_var, bg=BG, fg=MUTED_FG, anchor="w").pack(fill="x", padx=14, pady=(0, 8))

    list_frame = tk.Frame(win, bg=BG)
    list_frame.pack(fill="both", expand=True, padx=14)

    canvas = tk.Canvas(list_frame, bg=BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
    cards_frame = tk.Frame(canvas, bg=BG)
    cards_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=cards_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def switch_to(name: str, out_dir: Path):
        conversation.set_persona(out_dir, name)
        current_var.set(f"目前人格：{name}")

    default_card = {"name": f"{DEFAULT_PERSONA_NAME}（預設）", "description": "沒有特定角色設定的一般聊天模型。"}
    default_row = tk.Frame(cards_frame, bg=PANEL_BG, bd=1, relief="solid")
    default_row.pack(fill="x", pady=(0, 8), padx=2)
    tk.Label(default_row, text=default_card["name"], bg=PANEL_BG, fg=FG, font=("Segoe UI", 11, "bold"),
              anchor="w").pack(fill="x", padx=10, pady=(8, 2))
    tk.Label(default_row, text=default_card["description"], bg=PANEL_BG, fg=MUTED_FG,
              anchor="w", justify="left", wraplength=480).pack(fill="x", padx=10)
    ttk.Button(default_row, text="切換到這個角色",
                command=lambda: switch_to(DEFAULT_PERSONA_NAME, DEFAULT_OUT_DIR)).pack(anchor="e", padx=10, pady=8)

    cards = load_character_cards()
    if not cards:
        tk.Label(cards_frame, bg=BG, fg=MUTED_FG, justify="left", wraplength=480,
                  text="tranning/characters/ 底下還沒有任何角色卡。\n"
                       "用 character_model.py --build --name ... --description ... 建立。").pack(
            fill="x", pady=10, padx=2)

    for card in cards:
        row = _build_card_row(cards_frame, card, lambda n=card["name"]: switch_to(n, CHARACTER_CHAT_DIR))
        row.pack(fill="x", pady=(0, 8), padx=2)

    ttk.Button(win, text="關閉", command=win.destroy).pack(pady=12)
    return win
