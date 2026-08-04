## AI-module.
## 目錄(超連結)
01. [To do list](to_do_list.md)
02. [Error Log](ErrorLog.md)
## Rules
01. 不可離開與本專案其他的資料夾內容，本專案資料夾為: "C:\Users\roy\Documents\GitHub\AI-module" 以外的內容進行任意修改，除非我本人同意授權後才可以進行
02. 我會使用繁體中文與英文進行交流討論，當我輸入中文就用中文回應，英文就用英文回應，不可以問 A 答 B ，以免 token 用量的浪費
03. 有任何錯誤就輸出到 error log 當中，紀錄錯誤點好讓我可以檢查修正
04. 我會給予修正日誌，當修正好的結果確認功能沒問題後會讓你寫入到修正日誌當中，以後未來有遇到相似的問題可以快速排查
05. 不可以觸碰到 .env 檔案，這個由我自行手動進行輸入
06. 整個專案全部禁用其他 AI 模型或是 API ，而是我要自行設計 AI 模型，不管需要多高的算力等問題，要求就是可以在 RTX 4060 8G 為上限的算力能力進行處理，目前都是屬於私人使用，不考量對外公布等任何因素問題
07. 已變更 to do list 與 error log 的紀錄，變更於目錄(超連結)當中，可以看到有 markdown 檔案
## 需求
01. [x] 指令需要加上: /model, /init, /memory 等，並且需要選項框顯示，一樣與 home_screen.py GUI 一樣可以 tab 快速鍵入 — 見 to_do_list.md #17
02. [x] 角色的部分我要求能夠另外在/character的指令部分進入這個功能(另外的 GUI 介面，不是與 home_screen.py 同一個 GUI) — 見 to_do_list.md #17
03. [x] 一般情況下都是以一般的 AI 模型進行對話 — /model 預設值、/character 未選角色時都是 sinco 一般模型，維持這個預設行為
04. [x] 對於餵養的模型學習(machine learning)部分要的是使用一般的檔案，如音訊檔案、影片檔案、文字檔案等，我手頭上沒有足夠的 json 檔案可以給予模型進行訓練，所以 train_gui.py 需要改正適用檔案格式 — 見 to_do_list.md #17（tranning/dataset_import.py）

## 指令規則（由需求 #01/#02 落實為固定規範）
01. `/model`、`/memory`、`/init` 是真正的 Python 端指令（會改變狀態/寫檔），不是 app/command/*.md 那種「讀出來塞給模型當提示詞」的指令；新增這類指令時要沿用這個區分，不要混在一起
02. 指令的引數選項（例如 `/model` 的 auto/sinco/code）要能在輸入「/指令 」之後跳出選項框、用 Tab／上下鍵操作，跟指令名稱自動完成同一套機制（app/components/command.py 的 `ARG_SUGGESTIONS`）
03. `/character` 永遠開獨立的 Toplevel 視窗（app/components/character_browser.py），不可以把角色清單改成塞進 home_screen.py 聊天視窗本身的分頁或面板
