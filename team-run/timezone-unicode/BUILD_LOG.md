# BUILD LOG
## Deliverables
- `baisc/timezone_unicode.py`：標準庫 CLI，`--text`、UTC／本地 aware datetime、目前 offset、系統時區標籤與 Unicode 查碼。
  執行：`.\.venv\Scripts\python.exe -X utf8 baisc/timezone_unicode.py --text "台灣 中国 😀"`。
- `web/admin/frontend/src/components/action.py`：工具與個人化新增入口，獨立時區與 Unicode modal。
- `web/admin/frontend/src/components/style.css`：modal 文字與輸入框窄螢幕樣式。

## Per Sub-task
1. [High] CLI 已用 subprocess 檢查 `U+1F600`、`U+0301` 及 201 個 emoji 僅輸出 200 個碼位；Windows 測試子程序以 `-X utf8` 啟用一致編碼。
2. [High] 網頁以 `Intl.DateTimeFormat` 顯示 IANA 時區，`Date.getTimezoneOffset` 取得當下時差；PyScript unicodedata 處理逐碼位查詢，所有使用者輸出用 textContent。輸入限制 200 碼位；HTML maxlength=400 允許 200 個非 BMP 碼位，input 事件用 Python `[:200]` 精確截斷。無新增 fetch/storage/地區判斷或封禁。
3. [High] 從專案根執行 `.\.venv\Scripts\python.exe -m pytest web/backend/tests/test_frontend_chat.py web/admin/frontend/src/components/test_render_markdown.py -q`，兩檔共 8 passed；後者 mock 排程出現 4 個 coroutine 未 await 警告。單跑 test_frontend_chat.py 為 4 passed、無警告。action.py AST 語法檢查通過。瀏覽器驗證由主代理執行。

## Divergences from Plan
- none。研究結論已閱讀 `RESEARCH.md`，工具文字未聲稱可判定地區或國籍。

## Known Limitations
- Python 系統標籤不保證為 IANA 名稱；裝置／瀏覽器設定不是實際位置證明。
- Unicode 每列為碼位而非使用者感知字元，emoji／結合字可能占多列；採 ascii(char) 讓空白、換行、控制字元與非 ASCII 查碼一致可讀。
- 時間在開啟或輸入時更新，沒有背景計時器。
- 穩定 IDs：unicodeToolBtn、unicodeToolModal、timezoneInfo、unicodeInput、unicodeResult、unicodeCount、timezoneUnicodeClose。
- 尚待獨立 Reviewer 與主代理瀏覽器驗證，未寫功能完成驗收紀錄。

## 整合驗證
- 主代理實跑：pytest web/backend/tests/test_frontend_chat.py，4 passed，無 warning。
- browser-check.cjs 四時區 PASS：Asia/Taipei、Asia/Shanghai、Asia/Kathmandu（瀏覽器回報 Katmandu 別名）、America/New_York。
- 驗證 emoji、結合字、HTML 字串安全輸出、200 碼位限制、手機無水平溢出及 modal 關閉；每組工具期間 API 請求增量=0、pageErrors=[]。詳細 browser-results.json 與截圖存本目錄。
- CLI -X utf8 smoke 成功，U+4E2D/U+1F600/U+0301 正常。git diff --check 本輪產品檔通過。

## 使用者需求修正：撤下前端工具
- [High] 使用者明確要求「這個評判是要求不顯示在網頁前端」，已精準移除本輪 tools 入口、modal、unicodedata import、初始化掛載與專用 CSS，前端工具不列為交付功能。保留先前版型修改、所有舊註解、CLI 學習檔與研究。
- 未自行加入後端地區評判或帳戶封禁，等待主代理釐清內部觀察或帳戶限制規則。
- 撤下後驗證：`.\.venv\Scripts\python.exe -m pytest web/backend/tests/test_frontend_chat.py -q` → 4 passed，無警告。
