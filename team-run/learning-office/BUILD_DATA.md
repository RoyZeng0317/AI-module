# Builder：一般學習原創資料

## Changes

- 新增 `data/general_learning_v1.json`：208 筆、104 個原創雙語題組，繁中 104、英文 104；經人工近義合併後共 100 個分割群組。
- 每筆保留 `id/group_id/topic/language/prompt/reply/source_urls/originality_note`。預期前處理僅將 prompt/reply 送入模型，其餘供來源與群組追蹤，不當作問答本文。
- 主題筆數：crypto 32、Office 32、日常 18、學習 16、數學 16、科學 18、AI 16、溝通 16、電腦 18、理解與指令 24、地理比例尺 2。
- 全部逐題自編後人工檢視，不以替換名詞或複製問句擴增。雙語兩筆不是兩份獨立知識，故共用 group_id。
- 人工合併近義家族：`glv1_action_verification`（Office 能力與 AI 工具完成驗證）、`glv1_failed_action`（工具失敗與寫入權限）、`glv1_recall_spacing_methods`（回想、分散複習與三方法指令）。其餘每個雙語題組一群；父流程仍須檢查與既有來源的近義重疊。

## Provenance and scope

虛擬貨幣使用本輪 RESEARCH 已查證的兩個 Investor.gov 官方頁作風險脈絡：

- https://www.investor.gov/crypto-assets
- https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-alerts/crypto-asset-securities

沒有引用原文段落。數學示例、學習安排、模擬記錄步驟為本次原創教學設計；不把所有細節聲稱為上述頁面逐字支持。來源欄保留兩 URL 供父審查追溯。未填價格、買賣時機、平台推薦、收益承諾或美國法規套用台灣的說法。

Office 題目只教一般步驟、公式與能力邊界，不宣稱已建立檔案、能控制桌面或工具已接通。此輪沒有連到 Microsoft 官方文件，`source_urls=[]` 明示未額外取得外部來源；操作細節仍需在實際工具與版本上驗證。其餘基礎概念同為自編教材，沒有假造來源。

未讀取或複製獨立 eval 內容，沒有把評估問句當作撰寫材料；父流程負責對各獨立 eval 做排除與衝突隔離。沒有修改既有 pairs、checkpoint、路由、`.env`，沒有啟動任何正式訓練。

## Validation

以專案 `.venv` Python 直接讀 JSON 驗證成功：

- JSON 可解析；208 筆均有必要欄位及非空問答。
- ID、prompt 與移除非文數字後的正規化 prompt 均不重複。
- 英文問答不含中文字元；繁中問答均含中文字元。人工檢視英文以短句為主，必要術語仍需教學，不宣稱已做正式 CEFR 評量。
- 與 `data/pairs.json` 的 strip/casefold prompt 精確交集為 0；這不保證語意完全無重疊。
- 最長 prompt 75 字元、reply 191 字元；字元長度不等於 token 長度，訓練前仍須用實際 tokenizer 檢查。
- 檔案 SHA-256：`e344940d9360deff32f4c501fe138d52cc917395cc184ae094bf54976716bb74`。

## Review notes / error handoff

初次嘗試將完整約 29KB 原創內容放入單一 PowerShell command 時，工具拒絕呼叫（`CreateProcess ... rejected: blocked by policy`）。未執行、未寫檔。改成分批 apply_patch 落檔，再用短 Python 指令轉成標準 JSON 後成功；無專案執行期錯誤。父代理可將此工具限制併入 ErrorLog，本 Builder 未碰其他擁有者檔案。

這是訓練資料準備成果，不能推論模型已具備一般問答、投資判斷或 Office 操作能力。正式訓練、獨立評估與工具實作尚待父代理後續處理。
