# PLAN
## Restatement & Success Criteria
使用者要求：「我看了一下其實只是改變了網頁前端顏色而已，要求整個介面需要調整，希望能夠往gpt與claude那樣對齊」。本次重整 backend 實際服務的 admin 前端，採聊天產品常見的導覽、空白首頁與訊息工作區結構，不精確複製品牌。[High] 已由主代理確認 backend FRONTEND_DIR，且 action.py 為實際 DOM 建構來源。

1. 側欄以新增對話、歷史清單為主體；次要工具集中收整，歷史區可獨立捲動。
2. 頂列提供模型入口、目前對話標題與匯出功能，模型選擇沿用既有真實狀態與 API。
3. 無訊息時歡迎內容與輸入框組成置中首頁；送出或載入已有訊息後，訊息區成主體，輸入框固定主區底部且不蓋住訊息。
4. 桌機可真正收合側欄；手機以覆蓋側欄與遮罩開關，不擠壓訊息欄。
5. 保留對話新增、選取、重新命名、刪除、清除、Markdown 匯出、模型設定、角色／記憶／學習、通知、相機偵測、斜線指令及輸入法／換行／送出行為。
6. 保留使用者既有未提交修改與解說註解，不讀寫 .env，不修改專案之外檔案。

## Open Questions for the User
none。採用主代理確認的 admin 前端範圍；「往 gpt 與 claude 對齊」解讀為聊天導向資訊架構與互動方式。[Medium] 使用者未指定品牌版本、截圖或精確像素，故不宣稱品牌複製。

## Sub-tasks
| id | definition of done | depends on | risk |
| --- | --- | --- | --- |
| S1 | 首頁／對話狀態切換與輸入框位置正確，所有載入、送出、清空路徑一致 | none | 高 |
| S2 | 重組側欄、模型頂列與工具入口，既有處理函式和 DOM 參照不失效 | S1 | 中 |
| S3 | 桌機收合、手機遮罩與窄螢幕捲動成立 | S2 | 中 |
| S4 | 語法、既有功能契約與桌機／手機視覺檢查，獨立審查通過 | S1–S3 | 中 |
S1–S3 共用 action.py 與 style.css，採單一 Builder 順序修改，避免平行衝突；Reviewer 在完成後獨立檢查。

## Approach & Verification per Sub-task
- S1：以同一套 composer 與 shell 狀態 class 切換空白首頁、對話模式，不複製 textarea 或事件。先找出 append_message、清除、載入歷史等更新空狀態的位置，集中更新版型。驗證空白→送出→歷史載入→新增對話；若重建 DOM 造成 _dom 或事件失效，退回只搬移原節點／修改 class。
- S2：修改 _build_sidebar 與 _build_chat_card 的真實 DOM，讓主要導覽、歷史、次要工具分組；頂列模型入口沿用現有模型資料及設定機制，不新增路由。CSS 整理現有區段，避免堆疊僅改色覆寫。驗證既有 id、_dom key、handler 與功能入口完整；若模型事件依賴設定窗內節點，保留唯一 select 並安全調整歸屬。
- S3：桌機以 collapsed 狀態控制實際側欄寬度；手機以 open 狀態及 backdrop 控制覆蓋。切換同步 aria-expanded；遮罩可點擊關閉。檢查 1440×900 與 390×844、長歷史、長訊息和展開 textarea，無水平溢出，訊息／歷史可獨立捲動。若遮罩覆蓋 toggle 或 composer，修正層級與定位後才交付。
- S4：以不執行 PyScript 初始化的 Python 語法編譯檢查；可用既有 DOM mock 或瀏覽器預覽驗證實際 action.py 建構結果，不以 default.html 示範頁當成功證據。核對差異、功能入口與響應式截圖；如環境無法跑完整後端，清楚記錄實測範圍，不宣稱 API 全通過。

## Unknowns for Research
none。此工作使用本地既有程式與一般聊天介面設計，不需外部現況資訊；執行時程式依賴由 Builder 本地查核。[High] 已看到目前側欄、頂列、chat-empty、chat-messages、composer 的建構位置。

## Stop Conditions
實際 admin 前端有明顯結構與互動重整，成功條件全部通過且 Reviewer 無重大問題即結束。不新增後端功能、AI 模型、API、品牌圖樣或整站框架；不順手清理其他未提交變更。遇到實際錯誤依專案規則記入 ErrorLog.md；修正完成日誌需等待使用者確認，不自行標記功能已獲使用者驗收。
