# Builder UI handoff

## Scope / assumptions / confidence

S4 完成，僅改實際供應的 `web/admin/frontend/src/components/index.html`、`style.css`、`action.py`，並新增 `web/backend/tests/test_frontend_chat.py`。假設既有 PyScript 2024.1.1 CDN 與後端同源路徑維持；未增加套件或讀取 .env。信心：程式狀態與語法高、視覺需由 Integrator 瀏覽器確認。

## Changes / contracts

- 深色 sage 配色、易讀對話間距、歡迎標題與三個 starter prompt（投資能力／交叉熵／RAG），選取只填入輸入框，讓使用者確認後送出。
- 保留所有原有 DOM IDs 與設定、指令、攝影機、通知掛鉤。新增 `sidebar`、`sidebarToggle`、`inputStatus`；手機選單用 `.app-shell.sidebar-open` 切換並同步 aria-expanded。
- `_on_keydown()` 在 isComposing 或 keyCode 229 時直接返回，避免中文組字 Enter 送出或選單攔截。
- `send_chat()` 以 `_state['chat_busy']` 防止 await 初始化期間重送，finally 恢復按鈕、焦點、aria-busy 與提示；初始化錯誤顯示可重試訊息並保留輸入。既有 API 失敗則移除 pending 樣式並恢復原問題。
- `/api/chat` 可選 `trace` 字串或 dict，`_append_trace()` 用 textContent 放入摺疊 details，不解讀或推測「已上網」；僅最新當次回覆顯示，歷史 API 未保存 trace 因此不回填。
- Markdown http(s) 來源連結可點、新分頁加 noopener/noreferrer；仍先 escape，不接受 javascript URL。
- index 加 zh-Hant 與 PyScript 載入提示（初始化 build_ui 會移除）；載入失敗時仍有網路檢查說明。

## Validation

- `python -m py_compile web/admin/frontend/src/components/action.py`：通過。
- `.venv/Scripts/python.exe -m pytest web/backend/tests/test_frontend_chat.py -q`：2 passed。
- 回歸驗證重複請求防護、初始化失敗狀態／輸入復原、來源 URL 與 HTML escape。
- 未自行啟動瀏覽器；Integrator 正進行實際頁面驗證。無長時間訓練、無模型品質主張。

## Known limits

PyScript 首次載入依賴外部 CDN；載入提示不是無 Python 的替代聊天客戶端。一般多輪理解與網路內容正確性由共用後端負責；前端僅顯示實際回傳資料。

## Review revision

已處理 REVIEW.md MINOR：`_inline()` 使用同一次匹配辨識 Markdown link 與裸 http(s) URL，實際 `_with_source()` 的換行來源會轉為可點連結，避免重複包 anchor；仍先 HTML escape 並限制 http(s)，保留 noopener/noreferrer。新增直接擷取真實 tools.py helper 的測試，涵蓋含 query string 的 URL、既有 Markdown、javascript 前綴與引號注入。初次測試指出 trim 分號會截斷 HTML entity，已移除分號 trim 並重驗。整合者已確認桌面投資中英文回覆與 trace 顯示正常。

瀏覽器驗證另發現 `#cmdMenu.hidden=true` 被 `.chat-item-menu` 的 display:flex 覆蓋，收合後仍有 14px 空白條。加入全域 `[hidden] { display:none !important; }` 尊重 HTML hidden 語意；已檢查其他選單，對話項目選單以獨立 CSS class 管理，不受影響。補一項短 CSS 規則檢查；前端測試合計 4 passed。整合者已於 390×844 驗證手機選單展開／收合，視覺截圖位於 `team-run/frontend-mobile.png`。
