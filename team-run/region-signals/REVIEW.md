# REVIEW
## Verdict: PASS
[High] 無 BLOCKER 或 MAJOR。實際程式、反例測試及本機真實資料庫查詢符合 PLAN；不代表真實所在地分類準確率或使用者驗收。

## Success Criteria Check
- 北京時區初判 → met。明確別名白名單；UTC+8、Taipei、Kuala_Lumpur、Singapore 不當北京。
- IP 第二訊號及 MY 獨立處理 → met。Shanghai+MY 為 malaysia_timezone_conflict，CN 無北京時區為 cn_ip_only；其他國別與 unknown 不被文字改判 CN。
- 文字僅弱訊號 → met。簡體代表字、U+FF0C、指定詞彙只記錄命中；繁體共用字反例與各項單獨訊號測試通過。
- IP 可信來源與故障處理 → met。僅使用 request.client.host；body ip/country 與任意 X-Forwarded-For 不採信。缺庫、壞庫、私有/無效 IP 測試回 unknown。
- 前端不顯示 → met。只增加 payload client_timezone，Intl 失敗回 null；查閱瀏覽器三時區結果皆 visibleAssessment=false。
- 結果內部保存 → met。request.state、deque(maxlen=200)；無原始訊息/IP；無公開結果端點、reply/trace 注入或模型提示詞注入。
- 範圍及授權 → met。action 固定 observe_only，沒有國籍結論或封禁；DB-IP 署名、更新說明及下載校驗資料齊備。

## Issues
無待修 BLOCKER/MAJOR/MINOR。已記載限制：[High] 用戶可修改時區；IP 定位出口而非真實位置；ASGI 來源仍依賴可信代理部署設定。字表是有限啟發式，不能作完整簡繁或方言鑑定。這些限制已在 REGION_SIGNALS.md 清楚說明。

## What Was Independently Cross-checked
- Reviewer 獨立重跑三檔 pytest：50 passed，2 個既有第三方棄用警告。
- Reviewer 另以真實 MMDB 呼叫 assess_region：Shanghai+CN → cn_timezone_and_ip、Shanghai+MY → malaysia_timezone_conflict、Taipei+TW → other_country_ip、UTC+8+私有 IP → unknown。
- 靜態追查 chat_endpoint 資料流及 smart_reply_traced 呼叫，確認結果未傳給模型或公開回應；檢查實際 patch.object 對象使用 module.region_signals，以避開裸 import 與 package import 的兩個模組實例問題。
