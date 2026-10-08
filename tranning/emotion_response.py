"""Conservative local responses to explicit emotional statements.

These are reviewed phrase rules, not a learned emotion classifier. Ambiguous
negation, quotations, third-person reports and mixed feelings are left to the
normal chat path. A valence regressor cannot safely decide these by itself.
"""

import re
import unicodedata

_PREFIX = r"(?:我(?:今天|現在|最近|這幾天)?|(?:今天|現在|最近)我?|)(?:覺得|感到)?(?:真的|有點|有點兒)?"
_NEGATIVE = r"(?:心情(?:很|真的|有點)?(?:不好|很差|糟糕|低落)|(?:很|好|有點|真的)?(?:難過|傷心|低落|焦慮|緊張|孤單|委屈|生氣)|(?:很|有點|真的)?不開心|(?:很|好|真的)?想哭)"
_POSITIVE = r"(?:心情(?:很|真的)?(?:很好|不錯)|(?:很|好|超|真的)?(?:開心|高興|快樂))"
_RELIEF = r"(?:(?:已經|現在|其實)?(?:不|沒有)(?:那麼|很|再)?(?:難過|傷心|焦慮|緊張)|心情(?:好一點|好多)了|(?:已經|現在)?好多了)"
_BOUNDARY = r"(?:(?:我)?(?:現在|今天|暫時)?(?:不想(?:聊(?:天|這件事)?|說(?:話|這件事)?|回答)|想(?:自己|一個人)(?:待著|靜一靜))|(?:先)?(?:不要|別)問(?:我|了)?|不用安慰(?:我)?)"


def _matches(expression, text):
    return re.fullmatch(expression, text) is not None


def explicit_emotion(message: str) -> str | None:
    text = unicodedata.normalize("NFKC", message).strip()
    # Only act on a complete short statement, not emotion words mentioned
    # inside an explanation, a quote, a third-person story or a question.
    text = re.sub(r"[。.!！~～]+$", "", text).strip()
    clauses = [p.strip() for p in re.split(r"[，,。！!]+", text) if p.strip()]
    if not clauses:
        return None
    if len(clauses) > 2:
        return None
    first = clauses[0]
    if len(clauses) == 1 and _matches(_BOUNDARY, first):
        return "boundary"
    kind = None
    for category, expression in (("relief", _RELIEF), ("negative", _NEGATIVE), ("positive", _POSITIVE)):
        if _matches(_PREFIX + expression, first):
            kind = category
            break
    if kind is None:
        return None
    if len(clauses) == 2:
        # Respect an explicitly stated wish not to talk. Other continuations
        # may change the meaning, so do not classify a fragment in isolation.
        return "boundary" if _matches(_BOUNDARY, clauses[1]) else None
    return kind


_RESPONSES = {
    "negative": "你怎麼了？願意跟我說說發生什麼事嗎？",
    "positive": "聽起來心情不錯！有什麼開心的事想分享嗎？",
    "relief": "了解。想聊的時候再跟我說就好。",
    "boundary": "好，先不追問。想聊的時候再跟我說就好。",
}


def emotion_reply(message: str) -> tuple[str, str] | None:
    kind = explicit_emotion(message)
    if kind is None:
        return None
    return (f"明確情緒表達（{kind}）→ 本機句型規則回覆，非模型生成", _RESPONSES[kind])


def negation_signature(text: str) -> tuple[str, ...]:
    """Conservative fuzzy-match guard; not a general semantic parser.

    Preserve the immediate scope as well as the negative marker so matching
    cannot simply drop 不/沒 or move it from one predicate to another.
    """
    return tuple(re.findall(r"(?:不是|沒有|不要|不|沒|別|無)[\u4e00-\u9fff]{0,3}|\b(?:not|never|no|cannot|can't|don't|doesn't|isn't|aren't|won't)\b", text.casefold()))
