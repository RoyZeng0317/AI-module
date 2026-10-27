"""多輪訓練與推論共用格式；不修改tokenizer或保留跨聊天狀態。"""
from bpe_tokenizer import SEP

FORMAT_VERSION = "dialogue_v1"
RESPONSE_BUDGET = 60


def build_prompt(tokenizer, message, history=None, max_tokens=452):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("當輪訊息必須是非空文字。")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("上下文token預算必須為正整數。")
    pairs = []
    for pair in history or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(x, str) for x in pair):
            raise ValueError("歷史必須是使用者／助手文字配對。")
        pairs.append(tuple(pair))
    current = f"使用者：{message}\n助手："
    if len(tokenizer.encode(current)) + 1 > max_tokens:
        raise ValueError("當輪訊息超過上下文預算，請縮短內容。")
    while True:
        text = "".join(f"使用者：{user}\n助手：{assistant}\n" for user, assistant in pairs) + current
        ids = tokenizer.encode(text) + [SEP]
        if len(ids) <= max_tokens:
            return ids
        pairs.pop(0)
