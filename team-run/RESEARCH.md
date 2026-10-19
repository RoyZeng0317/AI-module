# Research handoff

## Findings

信心：高（本機程式與歷史記錄）；未啟動模型、訓練或讀取 .env，未驗證外站當前可用率。

1. **真正聊天模型與品質限制**：`tranning/chats.py:695` 一般分支呼叫 `transformer_chat.reply(message)`，不是文件概覽所說的 GRU。`transformer_chat.py:781` 直接取樣並回傳解碼文字，無問題覆蓋度、回覆相關性或低品質拒答。現有 `gpt_chat_runs/config.json` 是 8 層、384 維、512 context；ErrorLog #30 與其後紀錄明確記載英文亂答、語言切換與少量 pairs 過擬合。相似度目前只用於 code retrieval（SequenceMatcher，threshold=0.6），一般聊天沒有可信檢索機制。MC Dropout 多次同答也不代表答對。
2. **已證實的錯誤路由**：`chats.is_code_request()` 只要以「寫一個／寫一段／寫個」開頭就認定程式碼，所以「寫一個故事」會走 sinco-code。`tools._WEATHER_KEYWORD` 是任意位置的「天氣」，「你會查天氣嗎」也被當成真正天氣請求。搜尋規則將英文 `what is` 與中文「你知道」直接擷取後綴，容易將能力問答當成百科。隱含 knowledge lookup 失敗後回落未受控模型生成，仍會亂答。
3. **上網已存在，但不等於完整搜尋**：`tools.web_search` 用免金鑰 DuckDuckGo Instant Answer，`wikipedia_search` 用免金鑰 Wikipedia 摘要；兩者偏百科。`general_web_search` 為 Serper，需要 SERPER_API_KEY，而且執行時會 load_dotenv，不宜直接實測。`_lookup` 固定先 DDG → Wikipedia → Serper，因此最新新聞可能先得到過期百科。timeout 已有（搜尋 6 秒），但 JSON decode／型別不符可能未被 requests.RequestException 完整涵蓋。DDG/Wikipedia 成功回應通常只有文字，缺來源 URL；Serper 才附 URL。查詢候選會送 auto_learn 待審，不是自動完成訓練。
4. **Web 共用邏輯現況**：`web/backend/app.py:553` 確實共用 smart_reply_traced，force_mode/persona/character 有傳，但丟棄 trace，未傳 temperature（ChatRequest 若有該欄位也無效）；只有 nvidia 會建立 history，本機 Transformer 完全不使用歷史，這是既有模型訓練格式的限制。不要直接拼接歷史誤稱已解決多輪理解。
5. **實際前端**：`FRONTEND_DIR` 指向 `web/admin/frontend/src/components`，根 index.html 只載 style.css、PyScript 2024.1.1 與 action.py。action.py 動態建立主要 UI，不是 python.py/default.html。必留 root、box、chatLog/chatEmpty/chatForm/chatInput/chatSend、historyList、btnNewChat、modelSelect/modelBadge、settingsModal、stage/video/overlay/results，以及既有設定、登入、通知與指令事件。PyScript CDN 初次載入可能較慢；可加純 HTML loading fallback，初始化後移除。現有 style.css 已有 1080/760px breakpoint，須驗證手機側欄是否壓縮聊天；textarea Enter/Shift+Enter 與命令選單由 action.py 處理，避免 CSS/重建 DOM 破壞事件。

## Conflicts & Caveats

- README/AGENTS 舊概覽稱一般聊天 GRU；實際已是 Transformer，應以執行程式與 ErrorLog 為準。
- 公開免金鑰 HTML 搜尋 fallback 可以提供一般索引，但服務可能限流或 challenge；必須誠實失敗，不得把 challenge 頁當搜尋內容。無 live provider 驗證就不應宣稱任何供應者穩定可用。
- 序列字面相似度不是語意理解；否定詞、主體與數字只差一字也可能反轉問題。勿用低閾值近似檢索宣稱解決泛化。
- 尚未測試 checkpoint 真實輸出，因此可證實的是程式根因與歷史限制，無法證實一般人類等級問答。
- 此次搜尋指令有一次查錯不存在 index.html 路徑，以及 PowerShell rg wildcard 路徑錯誤；已改用正確檔案，非產品錯誤，無 production 修改。

## Plan Impact

- 最小修正優先：限制 code routing 須有語言／程式語彙；能力與身份問答先處理，區分討論工具與實際執行工具。
- 明確「最新／新聞／上網／搜尋」先走真正 general search，附可點來源；一般知識仍可百科 fallback。全鏈路捕捉 JSON decode、錯誤型別、空結果、timeout，失敗不可回落亂生成。
- 一般聊天可提供精確已知問答與保守拒答／詢問細節；若加覆蓋度 guard，使用獨立未見句測試及否定/數字反例，不能以單純字元相似度當真實信心。
- Web 加回傳 trace 並顯示路由／網路狀態，可讓使用者理解回答來源；若傳 temperature，保持預設相容，確認 Transformer 分支實際沿用策略。
- 前端改動以 action.py 與 style.css 為主；保留上述 DOM 掛鉤與既有功能。測試要涵蓋錯分故事、能力詢問、一般知識、最新搜尋、無網路、無 key、來源，以及各模式的回歸。

## 使用者實際案例補充

- 「你懂得股票投資嗎?」：屬能力／一般知識詢問，不應只因股票字樣觸發 TWSE 即時行情；回答可提供投資基本概念能力與限制，使用者若要即時價格才查行情。須檢查 _STOCK_KEYWORD 是否過寬。
- 「什麼是交叉商損失?」：高度可能是「交叉熵損失」錯字，但不得直接默認；可回「你指的是交叉熵損失（cross-entropy loss）嗎？若是，它衡量模型預測機率與正確答案的差距……」，明確呈現假設，不輸出無關詞條。
- 「名詞解釋什麼是RAG」：現有 anchored search patterns 不一定接受「名詞解釋」前綴，因此會掉一般生成。AI 專案情境下可將查詢擴成「RAG retrieval augmented generation 檢索增強生成」，並表明採此 AI 語境；不可無條件把 RAG 其他縮寫含義混入。
- 以上三筆應作為固定回歸案例，並加入相鄰反例（股票今日行情／cross-entropy 正確拼寫／RAG 非 AI 語境）避免僅硬編碼三句。
