"""NVIDIA 雲端 API 客戶端 —— 供 `/model nvidia` 手動選用的外部模型。

CLAUDE.md 規則 #06 預設仍是本專案自行從零設計、訓練的模型（sinco 一般聊天／
sinco-code 程式碼，見 tranning/chats.py）；這支檔案只在使用者透過
`/model nvidia` 明確切換時才會被 import／呼叫（見 tranning/chats.py 的
smart_reply_traced()），不影響任何預設路徑，也不會被其他模式間接載入。

需要環境變數 NVIDIA_API_KEY，可自行在系統環境變數或 lib/.env 設定（後者由
使用者自行手動建立與輸入，本檔案只在執行期用 python-dotenv 把它載入成
process 環境變數，不會建立、修改或印出 .env 的內容——CLAUDE.md 規則 #05
不可觸碰 .env，金鑰一律由使用者自行輸入）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("環境變數 NVIDIA_API_KEY 未設定，請先設定後再使用 /model nvidia")
        _client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    return _client


def nvidia_reply(message: str, history: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    """呼叫 NVIDIA 雲端 API（nemotron-3-ultra）取得回覆。

    回傳 (思考過程, 正式回覆) 兩段。請求已經帶 `enable_thinking: True` +
    `reasoning_budget`，串流的每個 chunk 除了平常的 `delta.content`（正式
    回覆）之外，推理模型還會多帶一個 `delta.reasoning_content`（模型的
    思考過程）——這個欄位原本完全沒被讀取，直接被丟掉，所以之前不管
    `enable_thinking` 有沒有開，使用者都看不到任何思考過程。這裡把兩段
    分開收集，呼叫端（chats._nvidia_reply()）再決定思考過程要不要顯示。

    把串流回應收集成兩個完整字串再回傳（不逐字印到終端機）——呼叫端
    （chats.smart_reply_traced()）是在背景執行緒跑，GUI／CLI 只在整段回覆
    完成後才更新畫面一次，跟其餘既有模式的呼叫慣例一致。

    `history` 是 [(使用者訊息, 這個角色的回覆), ...] 的順序清單（GUI 的
    Conversation.history／CLI 的 state["history"]，兩邊都在 /resume 時從
    session_store 還原回來）。跟 sinco 不同——sinco 是字元級小模型，訓練資料
    從沒教過它讀「歷史｜這句話」這種格式（見 conversation.py 的說明），塞歷史
    反而會把回覆拉走；nemotron 是正常的 chat 模型，本來就是靠 messages 陣列
    裡一問一答的 user/assistant 輪替吃上下文，不把 history 攤進 messages
    等於每次都當成全新對話，也是 `/resume` 之後 NVIDIA 模式看起來「沒有還原
    對話紀錄」的根因。
    """
    client = _get_client()
    messages = []
    for user_text, reply_text in history or []:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": reply_text})
    messages.append({"role": "user", "content": message})

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384},
        stream=True,
    )

    reasoning_parts = []
    content_parts = []
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)
        if delta.content:
            content_parts.append(delta.content)
    return "".join(reasoning_parts), "".join(content_parts) or "..."
