# REVIEW

## Verdict

REVISE（High）：目前有 1 個 MAJOR；請完成一輪定向修正後重驗。假設本次只修正可重現路由、搜尋與 UI，不以有限規則取代自建模型或宣稱母語／Claude/GPT 同等能力。未修改 production、.env 或 checkpoint。

## Success Criteria Check

- 一般問候與指定案例：部分通過。英文 `Can you explain stock investing?` 不連線且命中英文規則；交叉商錯字會要求確認；RAG 會標明 AI 語境。`懂股票投資嗎` 未命中能力規則，仍落生成，屬已公開的有限句型範圍。
- 明確知識問題不誤執行工具：未通過，見 MAJOR。
- 不相關回答與誠實限制：結構檢查及已觀察領域衝突有改善；未知但句法正常的偏題仍可能放行。程式、BUILD_CHAT 與 BUILD_LOG 都誠實揭露；不能宣稱已全面解決正常問答或語意品質。
- 搜尋與來源：公開無金鑰 fallback 實測取得來源，但相關性有限；失敗回覆及 timeout 路徑有測試。不能把「取得索引」寫成「找到正確 CrossEntropyLoss 文件」。
- 共用路由：通過，chats.py 保留一般 Transformer 生成、手動模式與角色路徑；backend 使用原共用入口並傳回真實 trace。多輪理解未假稱修復。
- UI：實際 backend 供應路徑正確，保留既有掛鉤；desktop 視覺依整合者證據，mobile 尚待整合者驗證。來源網址顯示有小問題見 MINOR。
- 回歸：整合者報告 152 tests pass；本審查獨立實測路由與 live 搜尋，不把單元測試數字當語言品質證據。先前灰階／特徵修改不在本輪重構範圍。

## Issues

1. **MAJOR — tools.py 的工具優先 gate 仍劫持明確定義問題。** 獨立實測 `台股的意思是什麼？` 進 TWSE 並回當日大盤數字，`天氣的意思是什麼？` 進 wttr.in（失敗時甚至回行情式錯誤），雖然兩句已由新增 `_KNOWLEDGE_QUESTION_PATTERNS` 明確支援。現行排除只查 `什麼是` 而非 `是什麼／的意思是什麼`，因此先被 weather/stock 攔截。請在即時工具之前共用明確定義意圖檢查，補中文前後置定義、真實行情與天氣反例回歸；同一輪也應處理 `Can you search the latest Taiwan news?` 被 capability gate 全部略過的明確執行請求，或明確限制支援語法，不可宣稱英文查詢路由完整。
2. **MINOR — 搜尋輸出的裸網址在 Web 不可點。** `_with_source()` 輸出獨立行裸 URL，但 action.py 新 `_inline()` 只將 `[label](https://...)` 轉連結；因此 builder 所述來源「可點」尚未實現於正常搜尋輸出。請增加安全裸 http(s) autolink 或由搜尋端輸出 Markdown link，補真實搜尋輸出格式測試；保持 HTML escape 與 noopener。

## What Was Independently Cross-checked

- 閱讀 AGENTS、PLAN、RESEARCH、三份 builder handoff、BUILD_LOG，以及 chats/tools/backend/action/index/style 實際 diff 與 chat_quality.py。
- 透過本機 `.venv` 呼叫 `public_web_search('PyTorch CrossEntropyLoss documentation', max_results=2, lang='en')`：確實回 PyTorch 官方首頁與 Get Started Locally；二者不是 CrossEntropyLoss 文件。與 builder 的有限相關性說明一致（High），不代表精準搜尋品質。
- monkeypatch `_lookup` 與 auto_learn candidate 寫入後驗證題型；英文投資能力不查網、RAG 擴詞且標明語境、cross-entropy 定義進知識查詢、中文最新搜尋設 fresh=True。上述定義題在 lookup 之前誤進即時 API 的反例也已重現（High）。
- `_inline` 與 `_with_source` 的格式契約已逐行交叉檢查；裸 URL 不會產生 anchor（High）。
