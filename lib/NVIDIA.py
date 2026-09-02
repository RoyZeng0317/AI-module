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

from lib.components.memory_store import list_memories

MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# /memory 存的規則／個性（見 lib/components/memory_store.py）平常只是給人用
# `/memory list` 查看的結構化清單，不會自動影響任何模型的回覆——sinco 是字元級
# 小模型，本來就沒有指令理解能力，塞了也不會生效；nemotron 是真正的 chat 模型，
# 才有能力靠 system message 理解並遵守文字規則。所以只有這裡（NVIDIA 雲端模式）
# 把這兩個分類的記憶現組成 system prompt，每次呼叫都夾帶最新內容——之後使用者
# 用 `/memory add business_rule ...`／`/memory add developer_preference ...`
# 新增或用 `/memory del <id>` 刪除，下一次 /model nvidia 對話會自動反映，不用
# 另外同步。
_SYSTEM_PROMPT_CATEGORIES = ("business_rule", "developer_preference")

_client: OpenAI | None = None


def _system_prompt() -> str | None:
    entries = [e for category in _SYSTEM_PROMPT_CATEGORIES for e in list_memories(category)]
    if not entries:
        return None
    lines = ["以下是使用者透過 /memory 設定、要求你每次回覆都必須遵守的規則與個性："]
    lines.extend(f"- {e['text']}" for e in entries)
    return "\n".join(lines)


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

    回傳 (思考過程, 正式回覆) 兩段。請求帶 `enable_thinking: True`，串流的
    每個 chunk 除了平常的 `delta.content`（正式回覆）之外，推理模型還會多帶
    一個 `delta.reasoning_content`（模型的思考過程）——這個欄位原本完全沒被
    讀取，直接被丟掉，所以之前不管 `enable_thinking` 有沒有開，使用者都看不到
    任何思考過程。這裡把兩段分開收集，呼叫端（chats._nvidia_reply()）再決定
    思考過程要不要顯示。

    **不要加 `reasoning_budget` 到 extra_body**：曾經加過 `"reasoning_budget":
    16384`（跟 `chat_template_kwargs` 同層），這個模型目前的後端（V2 vLLM
    model runner）不支援這個參數，非串流呼叫會回乾淨的 400（`ValueError:
    thinking_token_budget is not yet supported by the V2 model runner`），但
    串流模式下 NVIDIA 沒把詳細訊息傳回來，openai SDK 只會看到一個空泛的
    500 `Internal server error`，很容易誤判成金鑰或模型名稱有問題。已用真實
    金鑰實測確認：只留 `chat_template_kwargs: {enable_thinking: True}`（拿掉
    `reasoning_budget`）就能正常拿到回覆，`reasoning_content` 依然抓得到。

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
    system_prompt = _system_prompt()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
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
