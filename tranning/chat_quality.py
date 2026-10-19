"""Small, explicit dialogue rules and structural output checks, not confidence scores.

These rules do not establish semantic correctness or replace model training.
"""
import re, unicodedata


def normalize(text: str) -> str:
    return re.sub(r"[\s!?！？。.,，；;：:]+", "", unicodedata.normalize("NFKC", text).casefold())


_DIALOGUE = {}
for prompts, answer in (
    (("你好", "嗨", "哈囉", "您好"), "你好！今天想聊什麼，或需要我幫忙處理什麼？"),
    (("hello", "hi", "hey"), "Hello! What would you like to talk about?"),
    (("謝謝", "謝謝你", "感謝"), "不客氣！還有需要幫忙的地方嗎？"),
    (("thanks", "thank you"), "You are welcome!"),
    (("how are you", "how are you doing"), "I am ready to help. How are you today?"),
    (("ok", "okay", "ok got it", "okay got it", "got it"), "Great. Tell me if you need more help."),
    (("what are you doing now", "what are you doing"), "I am here to read your message and help you. What would you like to do?"),
    (("who are you", "what is your name"), "I am sinco, an AI assistant built for this project. I am still learning, so I can make mistakes."),
    (("what can you do", "can you speak english"), "I can help with simple chat, code questions, and web searches. My English is still limited. Please tell me what you need."),
    (("你是誰", "你叫什麼名字"), "我是 sinco，這個專案自行訓練的 AI 助手。我可以協助對話與資料查詢，但目前仍可能理解錯誤。"),
    (("你會做什麼", "你能做什麼", "你有什麼功能"), "我可以協助一般對話、程式碼問題與網路資料查詢。請告訴我具體問題；目前模型仍在訓練，無法保證每個問題都能正確回答。"),
    (("你好嗎", "你今天好嗎"), "我可以正常運作，謝謝關心。你今天過得怎麼樣？"),
    (("我很難過", "我今天很難過"), "聽起來你今天不太好受。願意說說發生什麼事嗎？"),
    (("我很開心", "我今天很開心"), "很高興聽到你心情不錯！發生了什麼開心的事？"),
    (("再見", "掰掰"), "再見，有需要時再來找我。"),
    (("bye", "goodbye"), "Goodbye! Have a good day."),
):
    _DIALOGUE.update((normalize(prompt), answer) for prompt in prompts)


def dialogue_reply(message: str) -> str | None:
    """Only exact normalized intents; never use fuzzy matching as understanding."""
    text = normalize(message)
    if not text:
        return "請輸入想詢問的內容。"
    if re.fullmatch(r"你(?:懂得|懂|了解|知道|熟悉|會)(?:股票投資|股票|投資)(?:嗎|么)?", text):
        return "我可以協助解釋股票投資的基本概念，例如股票、ETF、分散投資與風險。你想先了解哪一項？若要即時行情，請提供股票名稱或代碼並說明要查的日期。"
    if text in {"doyouknowaboutstockinvesting", "canyouexplainstockinvesting", "doyouunderstandstockinvesting"}:
        return "I can explain the basics of stocks, ETFs, and investment risk. Investing can lose money. What would you like to learn first? For current prices, please give me the stock name or code."
    if text in {"為什麼", "然後呢", "再說一次", "繼續", "那個呢", "它是什麼", "為什麼會這樣"}:
        return "請補充你指的是哪個主題或貼上要我解釋的內容，我才能接著回答。"
    if text in {"why", "continue", "what about it", "say it again"}:
        return "Please tell me which topic you mean, or paste the text you want me to explain."
    return _DIALOGUE.get(text)


def output_issue(message: str, reply: str) -> str | None:
    """Catch observable corruption only; a passing reply can still be wrong."""
    text = reply.strip()
    if not text or text in {"...", "…"} or "<unk>" in text or "<pad>" in text:
        return "empty or undecodable output"
    compact = re.sub(r"\s+", "", text)
    if re.search(r"(.{1,16}?)\1{3,}", compact):
        return "repeated output"
    if normalize(message) == normalize(reply):
        return "input echo"
    # Narrow regressions observed in this checkpoint: finance questions got
    # coding replies, and ML definitions got English grammar lessons. This is
    # a domain conflict check, not a general semantic relevance classifier.
    domains = {
        "finance": r"股票|投資|股價|ETF|stock|invest",
        "ml": r"交叉[熵商]|損失|RAG|機器學習|神經網路|cross.entropy",
        "grammar": r"主詞|受詞|補語|S\s*\+\s*V|英文文法",
        "code": r"程式邏輯|seq2seq|程式碼|python",
    }
    query_domains = {key for key, pattern in domains.items() if re.search(pattern, message, re.I)}
    answer_domains = {key for key, pattern in domains.items() if re.search(pattern, reply, re.I)}
    conflicts = {"finance": {"grammar", "code"}, "ml": {"grammar"}}
    for domain, unrelated in conflicts.items():
        if domain in query_domains and domain not in answer_domains and (answer_domains - query_domains) & unrelated:
            return "known unrelated topic pattern"
    if re.search(r"[\u4e00-\u9fff]", message) and not re.search(r"[\u4e00-\u9fff]", reply):
        return "unexpected language switch"
    if re.fullmatch(r"[\x00-\x7f]+", message) and re.search(r"[A-Za-z]", message):
        requested_chinese = re.search(r"(?:translate|in|to|into)\s+(?:traditional\s+|simplified\s+)?chinese", message, re.I)
        if not requested_chinese and (re.search(r"[\u4e00-\u9fff]", reply) or not re.search(r"[A-Za-z]", reply)):
            return "unexpected language switch"
    return None


def clarification(message: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", message):
        return "目前模型未能產生可用的回答。請補充主題與希望解決的問題；需要查資料時，也可以輸入「搜尋＋關鍵字」。"
    return "I could not give a clear answer. Please add more details, or use 'search' with a topic to find information online."
