# REVIEW

## Verdict

PASS（High，限本次工程修正範圍）：第一輪 MAJOR 與 MINOR 已修正，本輪未發現 BLOCKER／MAJOR。此結論不是一般語意正確、英文母語能力或 Claude/GPT 同等品質的驗收。第一輪保存在 REVIEW_ROUND1.md。

假設：交付維持自建模型、有限明示規則、可辨認網路來源與誠實失敗；不以此次修改宣稱權重或泛化能力已改善。未知句法正常但偏題的生成仍可能通過 guard，這項限制已明確公開，需另行語料、訓練與人工評測。

## Success Criteria Check

- 指定案例與路由：通過本次定向回歸。投資能力回覆不誤查行情；交叉商先請確認；RAG／交叉熵精確定義回本機已審核詞條並明示來源日期與非即時搜尋。
- 定義／即時區分：第一輪 MAJOR 已修。獨立以會直接失敗的 market/weather mock 驗證「台股的意思是什麼？」「天氣的意思是什麼？」不碰即時 API，改走知識主題。`Can you search the latest Taiwan news?` 實測 lang=en、fresh=True。
- 低品質與限制：保留一般 Transformer 生成與手動模式／角色，不以模板取代所有生成；guard 及搜尋過濾只代表有限規則或詞面檢查，不代表語意信心。交付必須保留此區分。
- 搜尋：公開無金鑰 fallback、timeout／驗證失敗回覆與來源保留。新增 RAG 污染拒絕可阻擋已重現的人物片段；詞面過濾可能拒絕同義詞或仍接受詞面相關但錯誤內容，不能稱可靠事實驗證。
- 共用入口／Web：通過程式檢查；backend 保留共用 smart_reply_traced 並回傳實際 trace；UI 不假稱每次已上網。
- UI：第一輪裸 URL MINOR 已修，真實 `_with_source` 格式已轉安全 anchor 並有 query string／注入反例測試。已親自看 390×844 截圖，文字與輸入區可讀；選單操作由整合者 live 驗證。截圖為 hidden CSS 修復前舊圖，空白條原因與新增 `[hidden]` 規則、測試相符。
- 測試：本審查獨立執行 tools/math/frontend 104 passed，evaluate runner 3 passed。完整整合與 classifier 原修改保留檢查交整合者最後執行。沒有進行真實模型訓練或24筆品質評測。
- 新人工驗收 runner：已讀實作。需要 --run；offline 明確禁用整個工具 router，故不是完整產品路由；--allow-network 才啟用工具。輸出 pending_human_review 與 semantic_score=null，未冒充品質分數。含原始案例，不能稱全部為未見 holdout。

## Issues

本輪無需阻擋交付的工程問題。第一輪兩項已關閉。

已知限制（非已完成品質目標）：有限精確句型、未知語意偏題、本機多輪 history 未訓練、免費搜尋相關性與可用率。新詞面檢查也可能拒絕短縮寫／同義改寫。請於交付如實說明並以人工評測追蹤，不將本次 PASS 寫成一般正常人類問答能力已達標。

## What Was Independently Cross-checked

- 重讀 BUILD_LOG Integration update、BUILD_SEARCH、BUILD_UI、BUILD_CHAT 新增驗收流程；檢查 tools/action/style 實際修改與 evaluate_chat.py。
- 獨立重現兩個定義反例、英文禮貌搜尋、精確本機詞條與污染拒絕；測試時停用候選資料寫入，未改 production／.env／checkpoint。
- RAG 詞條與原始論文摘要對照：論文明述檢索之非參數記憶與生成結合，支持目前概念性釋義。[原始論文](https://arxiv.org/abs/2005.11401)
- 交叉熵詞條與官方文件對照：對單一正確類別、無權重的基本情況，−log(p) 與頁面公式相符；程式 API 本身接 logits，詞條未誤教把 p 直接作 API 輸入。[PyTorch 官方文件](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- 本輪獨立跑 107 個定向測試全部通過。第一輪真實無金鑰搜尋只取得官方首頁／安裝頁，其有限相關性的判定仍保留；未把此次已審核詞條標成 live 搜尋。

## 實際模型評測補充

整合者已執行24筆 offline 現有模型推論，輸出 `team-run/chat_eval_offline.jsonl`，3筆網路題跳過；本審查抽讀輸出確認仍保留 semantic_score=null／pending_human_review，且可見不自然或偏題內容。此結果支持「正常人類問答尚未完成、未重訓」；本輪 PASS 僅工程 phase，不能對整體使用者語言品質目標宣告完成。offline 禁用 router，因此其知識題不能當正式線上路由效果分數。
