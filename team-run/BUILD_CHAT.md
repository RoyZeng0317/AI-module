# Chat Builder handoff

深度：standard；使用 Fable。狀態：完成 S2/S3 聊天範圍，可供獨立審查。

## FRAME
修正已確認的聊天路由錯誤、提供保守一般對話與可觀察的低品質檢查。不重訓 checkpoint，不加入外部模型。只改 chats.py、新 chat_quality.py 與聊天測試。

## DECOMPOSE / PLAN
1. 撰寫故事／感謝信不可只因「寫一個」而進 code；保留 Python/C++/C#/SQL 與排序法。
2. 精確已審核中英文問候、身份、能力、情緒簡短回應與股票能力問題；不採模糊字串相似度當信心。
3. 未知指代追問要求補充主題，不任意拼接歷史；其他未知輸入保留 Transformer 生成。
4. 空白、echo、重複、語言切換及已重現的狹窄互斥領域模式回到同語言澄清。

## EXECUTE / Deliverables
- `tranning/chats.py`：code route 需同時包含撰寫形狀與程式術語，加入英文撰寫句型；一般角色處理之後才套用一般對話與生成品質檢查。保留手動 nvidia/code 及角色回應路徑。空白訊息提早返回。
- `tranning/chat_quality.py`：明確標記規則輔助並非模型理解，內含精確正規化對話與有限輸出檢查。投資能力只說可解釋概念及請使用者選主題，不給交易建議；不查即時價格。英文回應採短句、A2+ 詞彙，`how are you?`、`ok, got it.`、`what are you doing now?` 有對應。
- `tranning/test_chat_quality.py`：故事／程式碼分流、否定相鄰反例、投資能力、雙語、未知繼續問句、保留一般生成、手動外部模式及來源已觀察偏題模式。
- `tranning/test_chats.py`：Transformer 路徑測試以未知一般句取代已知 hello，避免把規則命中誤當生成測試；移除測試 echo 輸出，因 echo 現在會正確被 guard 攔截。

## VERIFY / Per sub-task verification
`\.venv\Scripts\python.exe -m pytest tranning/test_chats.py tranning/test_chat_quality.py -q`

結果：45 passed，5.97 秒；僅既有 pytz utcfromtimestamp deprecation warning。包括現有短合成 GRU 訓練測試，沒有真實模型長訓練、下載或覆寫 checkpoint。

獨立反例：`用 Python 分析股票` 不會因答案有程式碼被視為財經偏題；`RAG 是用來回答英文文法嗎` 可談文法；英文明確要求 `Translate hello into Chinese` 允許中文。`蘋果是一種水果` 對一般整理資料問題，結構 guard 仍會放行，測試明確保留此限制，不假裝已解決所有語意偏題。

## Divergences / Limitations
- 比最初空白／重複檢查多加入有限的 finance→code/grammar、ML→grammar 模式，根據 BASELINE 的真實回歸；這仍不是通用語意相關性分類器。
- 聊天模板是精確已審核規則輔助，**不是**模型已學會正常人類理解或已達 Claude/GPT 等級；本次沒有更新權重。
- 無法保證未知且句法正常的回答相關或正確；需要更完整雙語語料、訓練與獨立品質評測。
- 非英文字母的其他語言能力沒有新增訓練，只提供目前中英文預設與誠實澄清。英文技術原文、來源引用等仍可能需要整合層保留語言意圖。
- 角色模式沿用原行為，沒有套用一般對話模板或本次 guard，以免改變既有 persona。
- 工具回覆由 tools builder 負責；一般規則在 route_reply 之後，因此 tools 必須對能力問答回 None。

## REFLECT / Conclusions
- [High] 此次可驗證修復是路由與已知回歸／結構檢查；45 項本地測試通過。
- [Medium] 精確常用中英文對話改善命中句型體驗；未見改寫仍取決於本地模型。
- [Low] 無證據表明完整語意能力或母語程度有提升，因此不可對外宣稱。

## Assumptions
工具 builder 會處理 RAG、交叉商／交叉熵與上網來源；一般 chat guard 不作知識來源判定。主整合者處理 ErrorLog、README 與完整跨模組測試。PowerShell 曾因 `tranning/test*` 路徑模式錯誤失敗，改為 `rg ... tranning -g 'test*py'`；沒有產品執行錯誤。

## 後續整合修正：英文股票能力詢問
FRAME/PLAN：補已確認的英文能力句缺口，只採精確正規化三句，不攔截股價與特定投資原因問題。
EXECUTE：`Do you know about stock investing?`、`Can you explain stock investing?`、`Do you understand stock investing?` 回全英文基本概念能力、風險與可能損失，不給買賣建議。
VERIFY：`python -m pytest tranning/test_chat_quality.py -q` → 33 passed，2.86 秒，只有同一 pytz deprecation warning；包含特定昨日虧損問題不匹配的反例。
REFLECT：[High] 三個精確能力句不會掉入生成；[Low] 不代表英文金融推理能力提升。假設與先前相同。

## 後續範圍：可重複人工品質驗收（非訓練）
FRAME：建立下輪訓練前後可重跑的獨立人工語意驗收基準；不新增產品回答規則、不把測試案例塞回訓練 pairs。
DECOMPOSE/PLAN：分開固定案例、顯式推論CLI、無模型測試；預設不連網且必須 --run 才執行推論。未知語意不以字串符合率冒充理解。
EXECUTE：新增 `data/chat_eval_cases.json` 24 筆中英案例，包含日常改寫、主體／否定、情緒、概念、三使用者例句、英文自然改寫及含 history/context_required 的多輪案例。案例有 id、prompt、language、expected_intent、review_criteria，網路案例明確標記 requires_network。三原始例句是已知回歸，**不能**宣稱24筆全為未見句或使用它們當純holdout準確率。

新增 `tranning/evaluate_chat.py`：以 auto 本地模式輸出JSONL，每筆包括原案例、route、reply、latency_ms、tools_enabled、status，以及 `semantic_review=pending_human_review`、`semantic_score=null`。捕捉每筆例外供人工查核，不自動給語意分數。預設offline時暫時替換該engine的route_reply為None，避免未標記案例意外觸發網路；明示網路案例直接skip。--allow-network才保留真實工具router。此offline路由刻意不同於完整產品，報告已標示tools_enabled，需比較完整產品時另行允許網路。

VERIFY：`python -m pytest tranning/test_evaluate_chat.py -q` → 3 passed，0.10秒，注入fake engine，驗證未加--run不執行、offline router禁用與網路案例skip、history原樣傳遞、JSONL人工審查欄位、顯式network開啟。**未執行24筆真實模型評估，未訓練、未連網、未修改chats.py或tools.py。**

待使用者／整合者選擇執行時，可用：
```powershell
.\.venv\Scripts\python.exe tranning/evaluate_chat.py --cases data/chat_eval_cases.json --output output/chat_eval_offline.jsonl --run
# 明確允許上網的完整路由評估：另加 --allow-network
```
CLI將載入現有本地checkpoint推論，耗時依環境與模型而定；不自動觸發訓練。每次輸出檔會覆寫，請每輪指定不同檔名保存比較。

REFLECT：[High] runner的輸出與安全開關已有fake測試；[Medium] 案例涵蓋本次關注面向但僅24筆，不能代表全面能力；[Low] 未跑實模型，因此沒有品質改善的評估結論。假設：人工評審逐筆讀提示、history、回答與criteria，特別區分規則命中、工具查詢與模型生成；既有history不使用的限制要誠實記錄。
