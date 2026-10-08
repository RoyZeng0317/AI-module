"""persona_chat.py 測試：不載入任何真實模型，generate 用假的函式代替。"""

import json
import random

import pytest

import persona_chat
from persona_chat import (card_examples, create_card, fallback_reply, humanize, match_example,
                          persona_reply, resolve_traits, score_candidate)

CARD = {
    "name": "周柯宇",
    "description": "陽光男大的壞學生，很會寵溺愛人，也特別喜歡抱抱",
    "traits": {"寵溺": 1.0, "溫柔": 0.9, "撒嬌": 0.8},
    "prompt": {"我好想你": "我也是", "你在忙嗎?": ["剛結束這邊的事情呢", "怎麼了?", ""], "你在幹嘛?": ""},
    "call_user": "寶",
}


def test_resolve_traits_merges_slider_scores_with_description_keywords():
    traits = resolve_traits(CARD)
    assert traits["寵溺"] == 1.0            # 滑桿分數比描述推得的 0.8 高，保留較高值
    assert traits["活潑"] == pytest.approx(0.8)   # 「陽光」→ 活潑
    assert traits["俏皮"] == pytest.approx(0.8)   # 「壞學生」→ 俏皮


def test_description_only_card_gets_traits():
    traits = resolve_traits({"name": "小晴", "description": "話少、有點冷漠，但其實很細心"})
    assert {"高冷", "體貼"} <= set(traits)


def test_card_examples_skips_empty_drafts_and_expands_lists(tmp_path):
    examples = dict(card_examples(CARD, legacy_path=tmp_path / "missing.json"))
    assert examples["你在忙嗎?"] == ["剛結束這邊的事情呢", "怎麼了?"]
    assert "你在幹嘛?" not in examples  # 空字串是使用者還沒寫完的草稿


def test_legacy_examples_only_used_when_they_mention_the_character(tmp_path):
    legacy = tmp_path / "character.json"
    legacy.write_text(json.dumps([{"prompt": "你是誰", "reply": "我是周柯宇"}], ensure_ascii=False),
                      encoding="utf-8")
    assert "你是誰" in dict(card_examples(CARD, legacy_path=legacy))
    other = {"name": "小晴", "description": "活潑"}
    assert card_examples(other, legacy_path=legacy) == []


def test_match_example_ignores_character_name_and_punctuation():
    examples = [("柯宇在幹嘛呢?", ["我在想你阿"]), ("周柯宇", ["我在呢"])]
    prompt, replies, score = match_example("你在幹嘛", CARD, examples)
    assert replies == ["我在想你阿"]
    assert match_example("柯宇", CARD, examples)[1] == ["我在呢"]
    assert match_example("今天股市怎麼樣", CARD, examples) is None


def test_persona_reply_uses_user_written_example_first():
    trace, reply = persona_reply("柯宇你在忙嗎", CARD, generate=lambda m: pytest.fail("不該呼叫模型"),
                                 rng=random.Random(0), examples=card_examples(CARD))
    assert reply in {"剛結束這邊的事情呢", "怎麼了?"}
    assert "範例" in trace


def test_persona_reply_picks_most_in_character_candidate_and_removes_ai_identity():
    outputs = iter(["我是 sinco，一個人工智慧聊天模型。", "乖，我陪你，別擔心。", "好的。"])
    trace, reply = persona_reply("今天好累", CARD, generate=lambda m: next(outputs), n_candidates=3,
                                 rng=random.Random(1), examples=[])
    assert reply.startswith("乖")
    assert "sinco" not in reply and "人工智慧" not in reply
    assert "第 2 個" in trace


def test_persona_reply_falls_back_when_model_not_trained():
    trace, reply = persona_reply("嗨", CARD, generate=lambda m: "Transformer 模型尚未訓練，請先執行",
                                 rng=random.Random(0), examples=[])
    assert "尚未訓練" not in reply and reply
    assert "備用短句" in trace


def test_humanize_terse_personality_drops_particles_and_extra_sentences():
    traits = {"高冷": 1.0}
    text = humanize("好啊啦~我們一起去吧！然後再去吃飯。", {"name": "阿冷"}, traits, random.Random(0))
    assert text in {"好啊", "好啊啦"} or not text.endswith("~")
    assert "吃飯" not in text


def test_humanize_replaces_sinco_self_name():
    text = humanize("sinco 會陪你的", CARD, resolve_traits(CARD), random.Random(0))
    assert "周柯宇" in text and "sinco" not in text


def test_score_candidate_prefers_trait_words_and_penalizes_identity():
    traits = {"溫柔": 1.0}
    assert score_candidate("沒關係，我陪你", traits) > score_candidate("我是AI語言模型", traits)


def test_fallback_reply_follows_top_trait():
    assert fallback_reply({"傲嬌": 1.0}, random.Random(0)).startswith("哼")


def test_create_card_from_description_does_not_overwrite(tmp_path):
    path = create_card("小晴", "個性活潑開朗，講話很直", call_user="你", characters_dir=tmp_path)
    card = json.loads(path.read_text(encoding="utf-8"))
    assert "活潑" in card["traits"] and "毒舌" in card["traits"]
    with pytest.raises(FileExistsError):
        create_card("小晴", "別的描述", characters_dir=tmp_path)
    assert persona_chat.load_card("小晴", characters_dir=tmp_path)["call_user"] == "你"


def test_load_card_finds_cards_saved_under_a_sanitized_filename(tmp_path):
    # character_model.build_character() 存成 Alice_Smith.json，但 UI 顯示/傳入的是原始名字
    (tmp_path / "Alice_Smith.json").write_text(json.dumps({"name": "Alice Smith"}), encoding="utf-8")
    assert persona_chat.load_card("Alice Smith", characters_dir=tmp_path)["name"] == "Alice Smith"


def test_load_card_never_reads_outside_the_characters_dir(tmp_path):
    characters = tmp_path / "characters"
    characters.mkdir()
    (tmp_path / "outside.json").write_text(json.dumps({"name": "outside"}), encoding="utf-8")
    (characters / "list.json").write_text(json.dumps([{"prompt": "x"}]), encoding="utf-8")
    assert persona_chat.load_card("../outside", characters_dir=characters) is None
    assert persona_chat.load_card("outside", characters_dir=characters) is None
    assert persona_chat.load_card("list", characters_dir=characters) is None  # 非 dict 的 JSON 不會噴錯


@pytest.mark.parametrize("query,prompt", [
    ("我今天不開心", "我今天開心"), ("我不難過", "我難過"),
    ("我不想聊天", "我想聊天"), ("我今天很傷心", "我今天很開心"),
    ("I am not happy", "I am happy"),
])
def test_example_match_rejects_opposite_emotion_or_negation(query, prompt):
    assert match_example(query, CARD, [(prompt, ["wrong reply"])]) is None


def test_persona_emotion_response_never_selects_opposite_example():
    trace, reply = persona_reply("我今天不開心", CARD,
                                generate=lambda _: pytest.fail("explicit phrase needs no model"),
                                examples=[("我今天開心", ["太好了，恭喜！"])])
    assert "你怎麼了" in reply and "規則" in trace
    assert "恭喜" not in reply


def test_persona_keeps_appropriate_user_written_emotion_example():
    trace, reply = persona_reply("我今天心情不好", CARD,
                                generate=lambda _: pytest.fail("example first"),
                                examples=[("我今天心情不好", ["怎麼了？我在聽。"])])
    assert reply == "怎麼了？我在聽。" and "範例" in trace


def test_persona_respects_boundary_with_character_alias():
    trace, reply = persona_reply("柯宇，我今天心情不好，不想聊", CARD,
                                generate=lambda _: pytest.fail("explicit boundary needs no model"),
                                examples=[])
    assert "不追問" in reply and "？" not in reply
