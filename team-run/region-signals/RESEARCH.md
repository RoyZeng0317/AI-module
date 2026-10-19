# RESEARCH
查閱日期2026-10-19。
- Unicode官方 https://www.unicode.org/charts/PDF/UFF00.pdf 將U+FF0C列為FULLWIDTH COMMA，非國家標記。[High]
- MaxMind https://support.maxmind.com/knowledge-base/articles/maxmind-geolocation-accuracy 說明VPN/代理時通常只能定位出口，非終端使用者真實位置。[High]
- DB-IP https://db-ip.com/db/download/ip-to-country-lite 提供October 2026 Country Lite免費MMDB與CSV，CC BY4.0需DB-IP署名，web app使用結果頁需link https://db-ip.com 。資料覆蓋與準確度有限，不能宣稱實際位置證明。[High]
- 官方本月MMDB壓縮包 https://download.db-ip.com/free/dbip-country-lite-2026-10.mmdb.gz 。主代理下載並檢查後存 web/backend/data/dbip-country-lite.mmdb。安裝maxminddb於專案venv。非AI API，不傳送使用者IP。
- 支援亞洲時區名稱應明確白名單，Asia/Shanghai及歷史相容名稱可列初判；Asia/Taipei、Asia/Kuala_Lumpur、Asia/Singapore或僅UTC+8不能等同北京。[High]
- 文字詞表為使用者指定測試線索，不是經驗證的方言分類模型；用語、簡體與全形逗號都可被引用或由海外使用者輸入，僅記錄命中。[High]
