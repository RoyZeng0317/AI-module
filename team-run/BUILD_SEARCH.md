# Search builder handoff

## Scope / confidence

只修改 `tranning/tools.py`、既有 `test_tools.py` 與新增 `test_tools_search.py`。未讀取／修改 `.env`；未使用雲端 AI、未啟動訓練。離線路由與格式回歸信心高；外站可用率及片段相關性信心有限。

## Changes

- 中文／英文能力問句（含 `你懂得股票投資嗎?`、`你會查天氣嗎`、`Can you explain stock investing?`）留給共用聊天；不誤執行行情／天氣工具。
- `交叉商損失` 明確請確認是否指交叉熵，不默默當成正確名詞。
- 支援名詞解釋前綴、`X 是什麼`、`Explain X`。單獨 RAG 詞義題將查詢擴成 retrieval augmented generation／檢索增強生成，回覆明說採 AI 領域假設；其他領域 RAG 不強改。
- 新增「上網搜尋／上網查詢」，「上網搜尋台股最新新聞」不會誤抓大盤指數（另有回歸測試）。
- 明確搜尋／最新／新聞走一般搜尋，不先用百科摘要；未取得可靠知識查詢結果直接回報連線／驗證／空結果可能原因，不掉一般生成亂答。
- Serper 保留，只讀 process 環境 `SERPER_API_KEY`。無金鑰時公開 DDG HTML 搜尋，空頁／challenge／失敗再走 Bing RSS。單次供應者 timeout 6 秒；來源 URL 獨立一行。
- DDG／百科加入來源 URL；常見 JSON 錯誤、頂層型別、Serper 結果型別不符容錯。TWSE 異常 JSON rows 改成 ValueError，不洩漏 AttributeError。

## Verification

` .\.venv\Scripts\python.exe -m pytest tranning/test_tools.py tranning/test_tools_search.py tranning/test_tools_math_logic.py -q `：**89 passed**。
新增 fixture 測試涵蓋能力／名詞錯字不連線、RAG 語境、英語回覆不含中文與 lang=en、fresh 禁止百科、無 key HTML、challenge → RSS、全部 timeout、Serper 錯型別及失敗不落回生成。既有測試因來源 URL 與能力路由契約改變更新預期；初次失敗皆已修正。

2026-10-18 本機 live smoke（plain requests、未讀 .env）：
- DDG HTML：`PyTorch CrossEntropyLoss documentation` 與「台灣 今日 新聞」均回 **202 challenge**，不當成有效答案。
- Bing RSS：GET 回 **200**；正式 `public_web_search('PyTorch CrossEntropyLoss documentation', max_results=2)` 成功取得 PyTorch 官方首頁及安裝頁 URL／片段。
- 這證明無 key fallback 可取得索引，**未證明每次相關性**；此次第一頁並非精確 CrossEntropyLoss 文件。輸出仍是原始搜尋片段／來源，不能宣稱完成可靠推理或即時新聞正確性。

## Integration notes / limitations

- 一般能力回覆應由 chats 端處理；此處回 None 是刻意保留單一共享對話流程。
- 這不是 native English 或一般人類問答訓練成果，只是有限中英路由、誠實失敗與來源改善。
- 最壞百科 fallback 依序多個 6 秒 provider，公開搜尋兩次 request；不要宣稱無延遲或網路永遠可用。
- 請整合者將 DDG 202 驗證／初次回歸測試失敗記錄於 ErrorLog；避免與其他人同檔寫入衝突。


## Review round 1 revision

已處理 REVIEW 的 MAJOR：
- 新增共享 `_is_definition_request()`，前置／後置定義語句先排除天氣與行情。`台股的意思是什麼？`、`天氣的意思是什麼？`、`什麼是台股？`、`什麼是天氣？`、`天氣是什麼？` 都走定義路徑；目前台股行情／台北現在天氣仍走原工具。
- 禮貌英語 `Can/Could/Would you search/find/look up ...` 正規化為搜尋，例 `Can you search the latest Taiwan news?` 使用 lang=en、fresh=True，不再被能力 gate 吞掉。
- 整合者瀏覽器發現 RAG 搜尋混入政治人物／Facebook，已新增保守 `_search_relevant()`：RAG 擴詞查詢要求可見片段出現 RAG／retrieval augmented generation／檢索增強生成；一般多詞查詢要求最多兩個非停用詞命中。DDG HTML、Bing RSS、Serper 與摘要／百科 fallback 都過濾。這只是詞面清理，不是語意理解或正確性信心。
- 根據整合者已核對的原始論文／PyTorch 文件，新增兩個精確本機詞條：RAG 與交叉熵損失（中英原創簡短釋義）。畫面與 trace 明說「本機已審核詞條；來源核對 2026-10-18，非本輪即時搜尋」。RAG 僅精確定義題且說明採 AI 語境；「遊戲中的 RAG」不強制套入。交叉商錯字仍先確認。
- 原始 RAG 定義來源：https://arxiv.org/abs/2005.11401；交叉熵來源：https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html。來源查核由整合者提供，本子任務沒有把搜尋片段當成詞條證據。

最新驗證：同前三份 pytest **100 passed**（新增定義優先／即時反例、英文禮貌搜尋、政治人物污染丟棄、精確本機詞條無網路測試）。

最新 live smoke：`名詞解釋什麼是RAG` 與正確交叉熵定義題均命中明示本機詞條；`Can you explain stock investing?` 留 shared chat。公開 RAG 擴詞 query 在此次網路回應中 **無可接受結果**，會誠實回無可靠資料，不再輸出人物資料。先前 Bing 能取首頁的實測只代表索引可用；新過濾會拒絕與多詞查詢不符的首頁片段。


## Final typo UX correction

「交叉商損失」改成明示尚不確定原意，接著「若是交叉熵」就提供簡短條件式解釋，直接重用已審核詞條及 PyTorch 來源。使用者若不是此意須補完整名詞／情境；不再承諾「請確認，我再解釋」，避免只有回「是」卻因尚未支援多輪上下文再次偏題。回歸測試確認假設語氣、公式、來源及不含該承諾；未擴充多輪記憶。
