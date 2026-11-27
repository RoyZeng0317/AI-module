# 網頁 XSS 防護

本次修補以輸出邊界為主：文字以 DOM `textContent` 顯示，有限 Markdown 先處理不可信文字及 URL，再建立 HTML。訊息、模型／工具輸出、對話標題、通知、偵測標籤與上傳檔名都視為不可信資料；從歷史 JSON 重新載入時沿用相同規則。

對話仍保存原始內容，不刪除合法 HTML／程式碼範例、不在儲存時做 HTML entity 編碼。API 回傳 `application/json`，瀏覽器端不可將整份回應直接放進 `innerHTML`。JSON 本身含 `<script>` 字串不代表會執行，安全性取決於最後使用它的 DOM 邊界。

## 回應標頭

主 FastAPI 與 Firebase Hosting 設定使用：

```text
Content-Security-Policy: base-uri 'self'; object-src 'none'; frame-ancestors 'none'; script-src-attr 'none'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: strict-origin-when-cross-origin
```

`script-src-attr 'none'` 在支援 CSP3 的瀏覽器阻擋 HTML 事件屬性；`object-src` 禁止外掛物件，`base-uri` 限制 base URL；frame 設定禁止其他頁面嵌入本服務。原始碼提及 `about:srcdoc` 是前端偵測預覽環境的備援，正式頁沒有依賴 iframe 的流程。此設定可能使外站 iframe 預覽不可用，直接開啟服務頁面仍可使用。

**主站 CSP 是部分防線，沒有全面限制 script 來源或 inline script。** PyScript 2024.1.1 的 bundle 預設載入 Pyodide 0.24.1，涉及動態 import、WASM，以及可選 worker。未經完整 runtime／網路驗證，不猜測加入 `script-src`、`default-src`、`worker-src` 或 `connect-src`，也不以 `unsafe-inline`／`unsafe-eval` 放寬來掩蓋問題。因此安全 renderer 及 URL 驗證仍是核心防線，不能只依賴 CSP。

獨立影像服務沒有外部 runtime，使用較嚴格政策：

```text
default-src 'none'; script-src 'self'; script-src-attr 'none'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'
```

`blob:` 僅允許影像預覽；腳本、樣式、輪詢與結果檔案均同源。上傳依實際影像解碼，不能靠偽造 `Content-Type` 將 HTML 當作圖片。輸出圖片重新編碼為 PNG，下載 JSON／CSV 保留對應 MIME；不將原始上傳檔案以 HTML 提供。

## 驗證與限制

`web/backend/tests/test_xss_security.py` 使用專案內暫存 conversation／session／usage 路徑，mock 模型後由 API 寫入、改標題、重新讀取惡意內容，確認字串無損且回應是 JSON；產出 `output/xss-tests/api-replay.json` 供實際 renderer／Chromium 回播。儲存測試與 DOM 執行測試是不同證據，不能以 API 測試代替瀏覽器驗證。

標頭測試包含根頁、靜態檔案、JSON、404、500、MOUNT_PREFIX redirect 與子路徑。影像測試包含惡意檔名、假圖片、PNG signature 與 MIME。完整證據見 `team-run/xss/`。

Firebase 標頭只有設定完成，尚未部署；反向代理可能改寫標頭，上線後應再檢查實際 HTTP 回應。完整 PyScript 啟動、Google 登入、付款、實際部署與不同瀏覽器相容性未由此次 API 測試驗證。未宣稱所有 XSS 永久免疫，也未改動登入／CSRF 等其他安全機制。
