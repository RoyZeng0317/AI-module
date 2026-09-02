"""Unit tests for lib/NVIDIA.py's system-prompt injection.

`/memory`（memory_store.py）平常只是給人看的結構化清單，不會自動影響任何
模型的回覆——sinco 是字元級小模型，沒有指令理解能力；只有這裡（/model nvidia
外部雲端模型）真的有能力靠 system message 理解並遵守文字規則，所以
_system_prompt() 把 business_rule／developer_preference 兩個分類的記憶現組
成 system prompt。這裡不測試真正打網路請求（沒有 NVIDIA_API_KEY 也要能跑），
用假的 client 攔截 chat.completions.create() 的呼叫參數即可。
"""

import lib.components.memory_store as memory_store
import lib.NVIDIA as nvidia


def _isolate_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_store, "MEMORY_PATH", tmp_path / "memory.json")


class _FakeDelta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, delta):
        self.choices = [_FakeChoice(delta)]


class _FakeCompletions:
    def __init__(self, captured: dict):
        self._captured = captured

    def create(self, **kwargs):
        self._captured["kwargs"] = kwargs
        return [_FakeChunk(_FakeDelta(content="回覆內容"))]


class _FakeChat:
    def __init__(self, captured: dict):
        self.completions = _FakeCompletions(captured)


class _FakeClient:
    def __init__(self, captured: dict):
        self.chat = _FakeChat(captured)


# ---------------------------------------------------------------------------
# _system_prompt()
# ---------------------------------------------------------------------------

def test_system_prompt_none_when_no_relevant_memories(tmp_path, monkeypatch):
    _isolate_memory(tmp_path, monkeypatch)
    assert nvidia._system_prompt() is None


def test_system_prompt_combines_business_rule_and_developer_preference(tmp_path, monkeypatch):
    _isolate_memory(tmp_path, monkeypatch)
    memory_store.add_memory("business_rule", "嚴禁批評反駁")
    memory_store.add_memory("developer_preference", "情緒要接住我")

    prompt = nvidia._system_prompt()

    assert "嚴禁批評反駁" in prompt
    assert "情緒要接住我" in prompt


def test_system_prompt_ignores_unrelated_categories(tmp_path, monkeypatch):
    _isolate_memory(tmp_path, monkeypatch)
    memory_store.add_memory("project", "跟規則無關的專案記憶")

    assert nvidia._system_prompt() is None


# ---------------------------------------------------------------------------
# nvidia_reply() — system message 有沒有真的被夾帶進 messages
# ---------------------------------------------------------------------------

def test_nvidia_reply_prepends_system_message_when_rules_exist(tmp_path, monkeypatch):
    _isolate_memory(tmp_path, monkeypatch)
    memory_store.add_memory("business_rule", "嚴禁批評反駁")
    captured: dict = {}
    monkeypatch.setattr(nvidia, "_get_client", lambda: _FakeClient(captured))

    nvidia.nvidia_reply("你好")

    messages = captured["kwargs"]["messages"]
    assert messages[0]["role"] == "system"
    assert "嚴禁批評反駁" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "你好"}


def test_nvidia_reply_without_rules_has_no_system_message(tmp_path, monkeypatch):
    _isolate_memory(tmp_path, monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(nvidia, "_get_client", lambda: _FakeClient(captured))

    nvidia.nvidia_reply("你好")

    messages = captured["kwargs"]["messages"]
    assert all(m["role"] != "system" for m in messages)


def test_nvidia_reply_keeps_history_after_system_message(tmp_path, monkeypatch):
    _isolate_memory(tmp_path, monkeypatch)
    memory_store.add_memory("business_rule", "嚴禁批評反駁")
    captured: dict = {}
    monkeypatch.setattr(nvidia, "_get_client", lambda: _FakeClient(captured))

    nvidia.nvidia_reply("第二句", history=[("第一句", "第一句回覆")])

    messages = captured["kwargs"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "第一句"}
    assert messages[2] == {"role": "assistant", "content": "第一句回覆"}
    assert messages[3] == {"role": "user", "content": "第二句"}
