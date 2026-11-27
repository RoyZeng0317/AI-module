# 內部地區訊號

`region_signals.py` 在 `/api/chat` 讀取原始文字、瀏覽器自行回報的 IANA 時區，以及 ASGI `request.client.host`，完全在本機查詢 IP 國別。沒有外部 AI/API 呼叫，沒有額外保存原文或 IP。

結果只存在 `request.state.region_assessment` 和 `app.state.region_assessments`（最近 200 筆、重啟清空），不回傳聊天、trace 或公開 API，不放進模型提示詞。這是 Python 系統規則，不是模型訓練、國籍推論或封禁。`action` 固定 `observe_only`。

`Asia/Shanghai` 等明確別名才是北京時區初判；`UTC+8`、臺北、新加坡與吉隆坡不等同北京。時區可修改。CN IP 是第二個獨立訊號，並非真實位置證明；VPN/代理定位通常只是出口。MY IP 有獨立狀態，北京時區與 MY IP 同時出現時保留衝突。其他國別或未知 IP 不因文字被改判 CN。

`region_rules.json` 可維護時區別名、有限簡體代表字、標點及使用者指定用語。代表字只是有限啟發式，不是完整簡繁判別；用語命中不是北京方言鑑定。標點包括 U+FF0C（，）、U+FF1F（？）、U+3001（、）、U+3002（。），繁體中文及其他地區使用者也會使用。四種標點統一存於 `text_signals.punctuation`，只列不同碼位，不依出現次數或種類重複加權，不影響地區狀態；原本內部 `fullwidth_comma` 欄位由此取代。文字三類都只記錄弱訊號。

## 運作與信任邊界

安裝 `web/backend/requirements.txt`，資料放 `web/backend/data/dbip-country-lite.mmdb`。缺庫、壞庫、私有/無效 IP 都回未知，聊天可繼續。客戶端 body 國別/IP 和應用層的 X-Forwarded-For 均不採信。

ASGI 的來源可能已被 Uvicorn 代理中介層改寫，因此部署時必須只信任實際受控代理。直接對外服務可使用 `uvicorn web.backend.app:app --no-proxy-headers`；代理部署應明確設定 `--forwarded-allow-ips` 為受控代理 IP，不能任意使用 `*`，且代理要覆寫而非直接接受外部轉送標頭。否則來源可被偽造，或只能查到代理所在國家。本程式不以客戶端自報值填補未知。

## 資料更新與授權

來源：[DB-IP Country Lite](https://db-ip.com/db/download/ip-to-country-lite)，按月更新，授權 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本地資料為 2026-10 版，未修改資料內容。前端側欄保留最小「IP Geolocation by DB-IP」來源連結，不顯示任何內部評估。資料準確度不保證。

更新時由上述供應商頁取得當月 `.mmdb.gz`，解壓後替換本機檔案並重啟後端。避免持續使用過期資料；若下載失敗保留舊檔，勿以錯誤頁覆蓋。下載資訊與 SHA256 見 `team-run/region-signals/database-download.json`。
