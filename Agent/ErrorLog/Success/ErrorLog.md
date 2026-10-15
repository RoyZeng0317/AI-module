## Error log
1. `frontend/src/components/machine_learning.py` — 環境中沒有安裝 `pandas`／`scikit-learn`／`joblib`，執行到 `import pandas` 就會 `ModuleNotFoundError`；就算裝了套件，程式讀取的 `data.csv` 在整個專案中也找不到，會接著 `FileNotFoundError`；另外 `data.csv`／`linear_regression_model.pkl` 都用相對路徑，換一個工作目錄執行就會找不到檔案；`model.predict([[170]])` 用原始 list 而不是 DataFrame，sklearn 會跳 `UserWarning: X does not have valid feature names`。**2026-07-09 已修正**：`pip install pandas scikit-learn joblib`；在同目錄新增 `data.csv`（60 筆 height/weight 合成資料，供教學示範用）；路徑改用 `Path(__file__).resolve().parent` 定位，不受執行時的工作目錄影響；預測改用 `pd.DataFrame([[170]], columns=["height"])` 對齊訓練時的欄名。用 `python -W error::UserWarning machine_learning.py` 實際執行驗證：從頭跑到尾沒有任何錯誤或警告（MSE≈22.3，170cm 預測體重≈69.1kg）。
2. `backend/app.py`（本次新增的攝影機物件偵測後端）— 原本用 `app.mount("/", StaticFiles(directory=FRONTEND_DIR), ...)` 把整個 `frontend/` 資料夾當靜態檔案公開，會讓 `frontend/src/.env`、`frontend/src/components/*.py`、`frontend/data/basic_data.sql` 全部可以直接用網址讀到（例如 `GET /src/.env`）。已改成白名單路由，只放行 `index.html`/`script.js`/`style.css`，其餘一律 404。已用 FastAPI TestClient 驗證：`/src/.env`、`/data/basic_data.sql` 回 404，`/`、`/style.css` 正常回 200，`/api/detect` 傳真實圖片仍正確回傳偵測結果。
3. `frontend/src/components/chats.py` 接上聊天功能時 — 本機環境完全沒設定 `ANTHROPIC_API_KEY`，直接呼叫 Anthropic SDK 會在送出 request 前於 `_validate_headers` 丟出 `TypeError`（不是 `AuthenticationError`），沒被 `/api/chat` 原本只接 `APIStatusError`/`APIConnectionError` 的例外處理接住，導致回傳未分類的 500 Internal Server Error，看不出真正原因。**2026-07-09 已修正**：`chat_reply()` 開頭先檢查 `os.environ.get("ANTHROPIC_API_KEY")`，沒設定就丟自訂的 `ChatNotConfigured`，`app.py` 接住後回傳 503 + 清楚訊息（`"ANTHROPIC_API_KEY is not set on the server"`）。用真的 uvicorn 伺服器 + curl 驗證：沒設 key 時回 503 而非 500；`backend/tests/test_app.py` 新增 3 個測試（空 messages 回 400、mock 過的 Claude 回覆正確帶回 200、沒 key 回 503），`pytest backend/tests/` 9 項全過。**⚠️ 2026-07-09 此做法已作廢**：你澄清聊天功能要自己開發模型，不能直接調用 Claude API，整個 `chats.py` 已改寫成從零打造的 seq2seq 模型（見 to-do #05），本條僅留作歷史記錄，`ANTHROPIC_API_KEY`／`anthropic` 套件依賴已從專案移除。
4. `frontend/src/components/chats.py` 改寫成自建 seq2seq 模型後，訓練迴圈 loss 直接變成 `nan`（見 to-do #05）。根因：`nn.CrossEntropyLoss(ignore_index=PAD)` 預設 `reduction="mean"`，當某個 batch 在某個時間步 t 的所有樣本都已經是 padding（例如同批裡剛好都是比 `max_len` 短很多的短回覆），該時間步就沒有任何有效 target，mean 會變成 0/0 → `nan`，一旦某一步出現 `nan`，整個序列的 loss、之後所有梯度都會被污染。**2026-07-09 已修正**：`criterion` 改成 `reduction="sum"`，自己統計每個 batch 的有效 token 數（`(tgt != PAD).sum()`），loss 除以有效 token 數而不是固定除以 `max_len`。修好後用合成假資料手動跑 40 epoch 驗證：loss 從 2.9548 平滑降到 0.0051，且能正確回覆訓練過的問答（"hello"→"hi there"、"bye"→"goodbye"），`test_chats.py` 3 項測試全過。
5. `app/components/home_screen.py`（見 to-do #09）本機環境沒裝 `markdown` 套件，直接執行 `python home_screen.py` 在 `import markdown` 那行就會 `ModuleNotFoundError`，跟這次麥克風功能無關、是既有問題，只是這次要實際啟動 GUI 驗證語音功能才發現。**2026-07-25 已修正**：`pip install markdown`。用 `timeout 6 python home_screen.py` 實際啟動驗證：程式完整跑滿 6 秒沒有噴任何例外（正常情況下 `windows.mainloop()` 本來就不會自己結束，用 timeout 強制關掉才確認得了「有沒有在載入階段就崩潰」）。
6. `tranning/tools.py` 的 `web_search()`（見 to-do #10/#12 的搜尋路由）— 你回報查「Tokenization」這類詞時，回覆在「referred to...」這種地方就斷掉，像對話說到一半。實測直接呼叫 DuckDuckGo Instant Answer API 重現：查有歧義的詞（同時撞到 "Tokenization (data security)" 跟 "Tokenism" 兩個條目）時，`AbstractText` 本身就是空字串，`RelatedTopics` 給的每一則預覽本來就是 DuckDuckGo 自己截斷成「...」結尾的短預覽（不是我們的程式碼在裁切——用 `repr()` 印出原始 API JSON 確認過），但 `RelatedTopics` 的 `FirstURL`（例如 `.../Tokenization_(data_security)`）帶著消歧義後的完整詞條名，拿這個更精確的詞重新查一次就能拿到完整的 972 字 `AbstractText`。**2026-07-30 已修正**：`web_search()` 拆出 `_fetch_abstract()`（純發 request 不做判斷）+ 新增 `_expand_truncated_related_topic(first_url)`：當 `RelatedTopics` 的文字以「...」結尾且有 `FirstURL` 時，解析 URL 最後一段（`unquote` + 底線還原成空白）當作更精確的詞重新查一次；查得到就回傳完整內容，查不到（網路錯誤或還是空的）就照原樣回傳那則截斷預覽，不會讓使用者什麼都看不到。用真實網路呼叫驗證：查「Tokenization」原本回傳 132 字的截斷預覽，修好後回傳完整 972 字說明；新增 5 項測試（`test_tools.py`，monkeypatch `_fetch_abstract` 模擬 AbstractText 直接命中／歧義詞展開成功／展開失敗時退回原預覽／RelatedTopics 本身沒截斷時原樣回傳），`tranning/` 全部 98 項測試通過。
7. `tranning/dataset_import.py` 的 `extract_video_frames()`（見 to-do #17，影片擷取影格給 road_sign_train.py/image_classifier_bnn.py 用）— 一開始用 `cv2.imwrite(str(frame_path), frame)` 存檔，實測發現 Windows 上的 OpenCV 對**非 ASCII 路徑**（這個專案的類別資料夾名稱本來就常常是中文，例如「停止」「讓路」）會**靜默寫入失敗**：`extract_video_frames()` 回傳的 counts 字典正常顯示有擷取到影格，但資料夾底下實際上完全是空的（不是丟例外，是真的什麼都沒寫進去），非常容易誤判成功。用真的中文類別資料夾名稱手動重現：`counts={'停止': 2}` 但 `glob("*.jpg")` 回傳 0 個檔案。**2026-07-28 已修正**：改用 `cv2.imencode()` 把影格編碼成 bytes 後，用 `Path.write_bytes()` 自己寫檔，繞過 OpenCV 內部對路徑的檔案 I/O 處理。用同樣的中文資料夾名稱重新實測：`counts={'停止': 2}` 且 `glob("*.jpg")` 也回傳 2 個檔案，檔案大小 > 0。新增迴歸測試 `test_extract_video_frames_writes_files_for_non_ascii_class_names`（直接斷言磁碟上的檔案數量，不能只信任 counts 的回傳值），`pytest tranning/` 74 項全過。（原編號為另一條分支 #6，見 to_do_list.md 下半段）
8. `app/components/test_command.py`（見 to_do_list.md #20，@ 檔案附加功能）— 這次要重跑 `test_command.py` 才發現、跟這次改動本身無關的既有環境缺口：環境沒裝 `beautifulsoup4`，`test_command.py` 在模組層級 `from chats import DEFAULT_OUT_DIR`，匯入鏈是 `chats.py -> tools.py -> from bs4 import BeautifulSoup`，缺套件會讓整份測試檔案在 collect 階段就 `ModuleNotFoundError`，一個測試都跑不到。**2026-08-05 已修正**：`pip install beautifulsoup4`。順手發現 `app/components/test_cli.py` 也有同類但**沒修**的缺口：少裝 `prompt_toolkit`（`cli.py` 需要，見 to_do_list.md #15 那次新增的 Tab 自動完成功能），這次維持現狀沒有安裝——超出這次 @ 附加功能的範圍，留給你之後真的要跑 `cli.py`/`test_cli.py` 時再處理。
9. `tranning/agent_workflow.py`（你先自己寫的 ReAct 展示流程，未 commit 就要我幫忙抓錯）三個問題：(a) `agent_node()` 把 Agent 自己產生的訊息包成 `HumanMessage`（應為使用者輸入），語意上錯了，應該用 `AIMessage`（模型自己的回覆）；(b) 檔案匯入了 `load_dataset`/`GRPOConfig`/`GRPOTrainer` 但整份檔案完全沒用到，環境沒裝 `trl`/`datasets` 時會直接 `ModuleNotFoundError`；(c) 環境也沒裝 `langgraph`/`langchain-core`，這兩個是這份 ReAct 流程真正用到的套件。同一批 `tranning/sft_agent_data.jsonl` 訓練資料裡 system 訊息打錯欄位名 `"conent"`（應為 `"content"`），且整個檔案是單筆 JSON 被排版成跨 8 行，不符合 `.jsonl`（一行一筆）格式。另外 `tranning/train_grpo.py` 用 `model="Qwen/Qwen2.5-7B-Instruct"` 當 GRPOTrainer 基礎模型，違反規則06（禁用其他 AI 模型/API，全部自建），也遠超 RTX 4060 8G 算力；且 `trl.GRPOTrainer` 綁定 HuggingFace `PreTrainedModel`/tokenizer 介面，`transformer_chat.py` 手刻的 `GPT`（`nn.Module`）沒有這層介面，換模型字串接不上，需要重寫一份不依賴 `trl` 的訓練迴圈。**2026-08-09 已修正**：`agent_workflow.py` 改用 `AIMessage`、移除未使用的 `trl`/`datasets` 匯入，`pip install langgraph langchain-core`；`sft_agent_data.jsonl` 改成正確的單行 JSON、修正 `conent`→`content`；`train_grpo.py` 整個重寫成不依賴 `trl` 的自訂 GRPO 訓練迴圈（`_sample_completion()` 溫度取樣、`_sequence_logprobs()` teacher forcing 算 log-prob、組內 reward 正規化當 advantage、對凍結的 reference model 算 KL 懲罰），套用在 `transformer_chat.py` 的自建 `GPT` checkpoint 上。用 `PYTHONIOENCODING=utf-8 python agent_workflow.py` 實際跑過：ReAct 流程正確跑完 Thought → Action → Tool → Final Answer；新增 `test_train_grpo.py`（5 項測試：reward function 單元測試、`load_prompts()` 解析、`grpo_train()` 端到端跑 3 步在 tiny 模型上不崩潰並正確存檔、checkpoint 不存在時回傳 `None` 而不是丟例外），`tranning/` 全部 175 項測試（扣除既有的、跟這次無關的 `fitz`/PyMuPDF 缺套件 collect error）通過。
10. `tranning/test_kicad_dataset_convert.py`（見 to-do #17）— 這次跑 `tranning/` 全部測試才發現、跟這次改動無關的既有環境缺口：環境沒裝 `fitz`（PyMuPDF），`kicad_dataset_convert.py` 在模組層級 `import fitz` 沒有 try/except 包起來，缺套件會讓整份測試檔案在 collect 階段就 `ModuleNotFoundError`，中斷整個 `pytest tranning/` 的收集（連其他測試都跑不了）。尚未修正，跟同類型的 `beautifulsoup4`/`prompt_toolkit` 缺口（見上面第8點）一樣先留著記錄，之後你真的要跑 `kicad_dataset_convert.py` 或做批次測試時記得 `pip install pymupdf`，或者暫時用 `pytest --ignore=test_kicad_dataset_convert.py` 繞過去。

11. `app/components/main.py` 第 6 行 — 你直接用 `C:\Users\Roy\.anaconda-desktop\micromamba\envs\cuda\python.exe`（你實際跑 GUI 用的環境，跟 `pytest` 用的專案 `.venv`是不同的兩份環境）執行 `main.py`，第一個錯誤是 `ModuleNotFoundError: No module named 'fit'`。根因是打字錯誤：程式碼下面第 47/50 行實際呼叫的是 `fitz.open(...)`/`fitz.Matrix(...)`（PyMuPDF 的正確模組名稱是 `fitz`），但第 6 行的 `import` 打成了 `import fit`，兩者對不上，且沒有任何地方真的用到 `fit` 這個名字，純粹是手誤。**2026-08-12 已修正**：`import fit` 改成 `import fitz`。修好這個 typo 後，接續在同一個 `cuda` 環境陸續發現連鎖缺套件（改一個、重跑，又跳下一個 `ModuleNotFoundError`，非一次性一次性列出）：`fitz`（PyMuPDF 本身也沒裝）→ `cv2`（`camera.py` 用的 OpenCV）→ `torchvision`（`tranning/OCR.py` 用的圖片前處理）；另外順手一併裝了 `markdown`/`sounddevice`/`beautifulsoup4`（`command.py`/`conversation.py`/`tools.py` 匯入鏈上都會用到，不然修完 `torchvision` 之後大概率會接著撞到）。這個 `cuda` 環境的 `torch` 是 `2.13.0+cpu`（`torch.version.cuda` 是 `None`，儘管環境名稱叫 `cuda`，目前實際上沒有裝支援 GPU 的 build），`torchvision` 也裝成對應的 `0.28.0+cpu`，避免版本不匹配另外炸開 —— **這點只是誠實記錄現況，沒有動它**：要不要換成真的支援 CUDA 的 `torch`/`torchvision` build（關係到 Rule 06「RTX 4060 8G」算力預算能不能真的用上 GPU）是你的決定，這次沒有問過你就自己不會擅自換。用 `timeout 6 "<cuda env>/python.exe" main.py` 實際驗證：修完全部缺口後完整跑滿 6 秒沒有再噴任何例外（GUI 視窗正常啟動並停留在 `mainloop()`）。這次的套件安裝只做在 `cuda` 這個環境，沒有動 `.venv`（`pytest` 用的環境已經在同一天稍早的另一項任務裝過 `openai`/`markdown`，見 to_do_list.md #21），兩份環境目前裝的套件組合不完全一致，屬於已知現況。

12. `lib/main.py`（`app/components/` 搬到 `lib/components/` 之後，第 15 行起改成 `from lib.components.camera import CameraPanel` 這種絕對匯入）— 你用完整路徑直接執行這支檔案（`<cuda env>/python.exe c:/.../lib/main.py`），噴 `ModuleNotFoundError: No module named 'lib'`。根因跟第 11 條的 `import fit` 打字錯誤不同，純粹是**執行方式**問題：Python 直接執行 `.py` 檔案時，只會把該檔案所在資料夾（`lib/`）塞進 `sys.path[0]`，不會是專案根目錄 `AI-module/`，所以找不到 `lib` 這個套件本身。CLAUDE.md「常用指令」原本就寫了正確跑法是在專案根目錄下 `python -m lib.main`，但直接用完整路徑執行（例如 VS Code「執行 Python 檔案」按鈕，或像這次一樣手動貼路徑）仍然很容易誤觸。**2026-08-13 已修正**：`lib/main.py` 在原本三行 `sys.path.insert` 之後、`from lib.components...` 之前，加一段保險：偵測 `__package__` 是空字串（代表不是用 `-m` 啟動），就把專案根目錄（`lib/` 的上一層）動態塞進 `sys.path`，這樣即使直接用完整路徑執行也能正常匯入。用完整路徑直接執行驗證：`timeout 5 python lib/main.py`，不再出現任何 `ModuleNotFoundError`，5 秒後被 timeout 強制關閉（exit code 124，代表卡在 `mainloop()` 正常運作中，不是啟動階段就崩潰）。**同日追加**：你接著回報 `lib/components/cli.py` 也用一樣的完整路徑跑法（`<cuda env>/python.exe c:/.../lib/components/cli.py`）噴出完全同一種 `ModuleNotFoundError: No module named 'lib'`（第 43 行 `from lib.components.function import read_as_chat_content`），根因相同，補上同一招保險（`PROJECT_ROOT_DIR` 算出來後，`if not __package__` 才插進 `sys.path`）。用 `python lib/components/cli.py --help` 直接完整路徑驗證：正常印出 argparse 的說明文字，不再噴任何例外；`pytest tranning/test_chats.py lib/components/test_cli.py` 24 項全過。

14. `lib/components/cli.py` 第 19 行 `COMMAND_DIR = APP_COMPONENTS_DIR.parent / "command"` — 這次打包 `nova.exe`（見待辦：終端機 `nova` 指令 + PyInstaller 輕量啟動器）過程中讀程式碼才發現、跟這次改動無關的既有 bug：`APP_COMPONENTS_DIR` 是 `lib/components/`，`.parent` 是 `lib/`，所以 `COMMAND_DIR` 實際指向 `lib/command`，但 `app/command/*.md`（`rules.md`、`BugFix.md`、`Architecture.md`…）才是真正放指令提示詞內容的地方，`lib/command` 這個資料夾根本不存在。影響：`_markdown_commands()` 會回傳空清單，`/help` 看不到任何 `app/command/*.md` 對應的斜線指令，這些指令實際上也永遠打不到（`process_input()` 找不到 `lib/command/xxx.md` 一律回「未知指令」）。尚未修正——不在這次打包任務範圍內，先記錄起來；要修的話應該是 `COMMAND_DIR = PROJECT_ROOT_DIR / "app" / "command"`。
13. `lib/NVIDIA.py` 的 `nvidia_reply()`（`/model nvidia` 專用）— 你回報「看不到 NVIDIA 的思考過程」。實際檢查後發現：request 裡其實已經帶了 `extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384}`，NVIDIA API 也确实會在串流的每個 chunk 多回傳一個 `delta.reasoning_content`（模型的思考過程，跟正式回覆的 `delta.content` 是分開的兩個欄位），但原本的 `for chunk in completion` 迴圈只讀 `chunk.choices[0].delta.content`，`reasoning_content` 完全沒被讀取就直接丟掉了——不是 API 沒回傳，是程式碼沒收。**2026-08-13 已修正**：`nvidia_reply()` 改成同時收集 `reasoning_content`／`content` 兩段，回傳型別從 `str` 改成 `tuple[str, str]`（思考過程, 正式回覆）；`tranning/chats.py` 的 `_nvidia_reply()` 跟著改回傳型別，`smart_reply_traced()` 的 `force_mode == "nvidia"` 分支把思考過程接在原本的「已手動切換為 NVIDIA 雲端模型…」說明後面（`trace` 多一段「思考過程：」），思考過程是空字串時（例如例外分支、或模型這次沒有輸出思考內容）不會多印出空段落。`tranning/test_chats.py` 更新既有的 nvidia 測試 + 新增一項「無思考過程時 trace 不該出現『思考過程』字樣」的測試，`pytest tranning/test_chats.py -k nvidia` 2 項全過。

15. `build/pyinstaller/nova.spec`（`/model nvidia` 切換後執行噴 `No module named 'dotenv'`）— 一開始以為是環境缺套件，但 `.venv` 裡其實已經裝了 `python-dotenv 1.2.2`，根因不在這裡。實際原因是 `nova.spec` 用相對路徑字串指定打包來源（`pathex=['tranning']`、`Analysis(['../../lib/components/cli.py'], ...)`），這種寫法會依打包指令執行時的工作目錄而定；`build/pyinstaller/nova/warn-nova.txt` 第 237 行留有 `missing module named NVIDIA - imported by chats (delayed)`，代表 PyInstaller 靜態分析當初就沒找到 `lib/NVIDIA.py`，連帶它依賴的 `dotenv`／`openai` 也沒被收進打包範圍。執行期能吃到 `lib/NVIDIA.py`（因為 `tranning/chats.py` 的 `_nvidia_reply()` 會先手動 `sys.path.insert()` 指到磁碟上真實存在的 `lib/` 資料夾，讀到的是原始碼檔案本身，不受打包內容限制），但檔案裡面 `from dotenv import load_dotenv` 這行在凍結的 exe 環境裡就找不到模組了——跟 ErrorLog 第 11-12 條「執行目錄不同導致 ModuleNotFoundError」是同一種根因換了個地方發生，只是這次不是使用者執行方式的問題，是打包腳本本身依賴 cwd。**2026-08-15 已修正**：`nova.spec` 改用 PyInstaller 內建的 `SPECPATH`（永遠指向 `.spec` 檔案自己所在資料夾，不受呼叫時 cwd 影響）組出 `PROJECT_ROOT_DIR`/`TRANNING_DIR`/`LIB_DIR`/`CLI_SCRIPT` 絕對路徑，取代原本的相對字串；`hiddenimports` 明確加上 `'NVIDIA'`，並用 `collect_submodules('dotenv')`、`collect_submodules('openai')` 一併收進去。用 `.venv` 的 PyInstaller 清快取重新打包驗證：新 exe 238MB（跟原本 230MB 量級一致，`torch` 等核心依賴都正常包進去），`warn-nova.txt` 不再出現 `NVIDIA`/`dotenv`/`openai`/`torch` 任何一項缺失警告；複製到專案根目錄實測 `/model nvidia` → `你好`，成功拿到 NVIDIA 模型的正式回覆與思考過程，沒有再噴任何模組錯誤，確認無誤後已覆蓋掉根目錄原本的 `nova.exe`。

16. `build/pyinstaller/nova.spec` 第 48 行 `upx=True` — `nova.exe` 執行到一半（觸發 `rich` 的延遲載入，例如 `console.status()`）噴 `zlib.error: Error -3 while decompressing data: incorrect header check`（`pyimod01_archive.py extract()`），接著整個程式被 `[PYI-32336:ERROR] Failed to execute script 'cli'` 終止。不是缺套件（缺套件是 `ModuleNotFoundError`，不是 zlib 錯誤），是 exe 內嵌的 PYZ 壓縮封存區塊本身壞掉。根因是這支 exe 用 `upx=True` 打包，而專案內嵌了 `torch` 等巨大 DLL（onefile exe 達 238MB 量級）——UPX 對這種已經高度優化過的大型二進位檔案本來就容易產生不穩定的壓縮結果，是 PyInstaller 官方文件明確警告過的地雷（尤其搭配 numpy/scipy/torch），這次剛好在第 15 條打包完、驗證過能跑之後，檔案又在某個時間點壞掉，研判是 UPX 壓縮出來的區塊本身就脆弱，容易被之後任何一次讀寫（含防毒軟體即時掃描）打壞。**2026-08-15 已修正**：`nova.spec` 的 `upx=True` 改成 `upx=False`，用 `.venv` 的 PyInstaller（`python -m PyInstaller build/pyinstaller/nova.spec --distpath build/pyinstaller/dist --workpath build/pyinstaller --noconfirm --clean`）重新打包，新 exe 238,294,108 bytes（跟舊的 238,285,667 bytes 幾乎同量級，代表 UPX 原本對這個 exe 也沒省到什麼空間，純粹只有增加壞掉的風險）。用 `build/pyinstaller/dist/nova.exe --help` 跟 `printf '你好\nexit\n' | nova.exe` 兩種方式實測驗證：`--help` 正常印出說明文字；實際對話時觸發 `rich` 的 `console.status()`（正是舊 exe 崩潰的同一段程式路徑）能正常印出彩色框線與 sinco 的回覆，沒有再噴任何 zlib 錯誤。確認無誤後已覆蓋掉根目錄原本壞掉的 `nova.exe`。

16. `cuda` micromamba 環境的 `torch`/`torchvision`（見 ErrorLog 第 11 條遺留的已知現況）— 當時記錄「環境名字叫 cuda，但裝的其實是 `torch 2.13.0+cpu`／`torchvision 0.28.0+cpu`，沒有真的吃到 GPU 算力」，一直沒換。這次要規劃語音模型訓練管線（自建 TTS，符合 Rule 06），CPU-only 會讓訓練時間變得不可行，所以動手換掉。先用 `nvidia-smi` 確認實際硬體：GPU 是 **RTX 5060 Laptop（8GB VRAM）**，不是 CLAUDE.md 原本寫的「RTX 4060」——顯示卡型號跟文件不一致，但同樣落在「8G 顯存」這個算力上限內，Rule 06 的預算沒有變。驅動版本 591.91 支援到 CUDA 13.1；用 `pip index versions torch --index-url https://download.pytorch.org/whl/cu130` 查證後，確認 PyTorch 官方索引剛好有跟原本裝的 CPU 版完全相同版本號的 GPU 版（`torch 2.13.0+cu130`、`torchvision 0.28.0+cu130`），可以同版本置換不用連帶處理其他套件的相容性問題。**2026-08-15 已修正**：`cuda` 環境用 `pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130 torchvision==0.28.0+cu130` 换成 GPU 版。驗證：`torch.cuda.is_available()` 回傳 `True`、`torch.cuda.get_device_name(0)` 正確顯示 `NVIDIA GeForce RTX 5060 Laptop GPU`；實際在 GPU 上跑一次 2000x2000 矩陣乘法（`torch.randn(..., device='cuda')` + `torch.cuda.synchronize()`）成功執行，確認不是只有「偵測到」而是真的能算；換完 torch 後用 `timeout 8 python lib/components/cli.py --help` 重新驗證 CLI 仍正常啟動、不受影響。`.venv`（`pytest` 用的環境）維持原本的 CPU 版沒有動，兩份環境套件仍不完全一致，屬於刻意保留的既有現況（見 ErrorLog 第 11 條），只是這次縮小了差距（版本號已對齊，只差 cu130 vs cpu 這個 build 差異）。

17. `web/frontend/src/components/action.py`（你回報瀏覽器噴 `blob = await new Promise(lambda resolve: cap_canvas.toBlob(...))` 的 `SyntaxError`，以及 `document`/`window`/`js` unknown import symbol）— 檢查後發現這支檔案本機端第373~380行**已經**用 `Promise.new(lambda resolve, reject: ...)` 的正確 pyodide 寫法修過（連同詳細註解都已經在檔案裡），代表你貼的那個 `new Promise(...)` 版本的錯誤不是來自本機這份原始碼，研判是 `前端部屬scp.md` 流程 scp 到 Pi5 上的**舊版**檔案還沒同步；`document`/`window`/`js` unknown import 是 Pylance 靜態分析看不到 Pyodide 執行期才注入的模組，純編輯器警告，不影響瀏覽器實際執行。**尚未完全驗證**：你提到要把這個橋接到 Pi5，順勢把物件偵測從「每 500ms 一次 HTTP POST `/api/detect`」改成「開一條 `/ws/detect` WebSocket 連線、相機開著就一直用同一條連線送畫面收結果」——`web/backend/app.py` 新增 `@app.websocket("/ws/detect")`，`action.py` 新增 `_open_detect_socket()`/`capture_and_send()`，`DETECT_WS_URL` 由 `BACKEND_URL` 自動換算 ws/wss。已用 `ast.parse` 驗證兩支檔案語法都過、`pytest web/backend/tests/` 新增的 2 項 WebSocket 測試（`test_detect_ws_endpoint_streams_detections_over_one_connection`、`test_detect_ws_endpoint_reports_bad_frame_without_closing`）都通過，但**沒有真的開瀏覽器 + 實體 Pi5 驗證過**（這裡沒有瀏覽器/Pi5 環境可以跑）——你重新 scp 更新檔案到 Pi5、實際開相機測試沒問題後，記得回報一下讓我把這條寫進「已修正」。另外跑測試時**順便發現一個跟這次改動無關的既有 bug**：`web/backend/app.py` 的 `FRONTEND_DIR = BACKEND_DIR.parent / "frontend"`（也就是 `web/frontend/`）跟 `PUBLIC_FILES = {"index.html","script.js","style.css"}` 這組靜態檔案路由，指向的其實是 `前端部屬scp.md` 提到的「另一個 Firebase Hosting 用的 script.js 版本」，但 `web/frontend/` 底下目前根本沒有任何檔案（`Get-ChildItem` 確認過是空的，PyScript 版三個檔案實際在 `web/frontend/src/components/`）——`test_index_served`／`test_public_assets_served` 這兩項測試因此本來就會失敗（用 `git stash` 驗證過在我這次改動之前就已經是失敗的），跟這次 WebSocket 改動無關，先記錄起來，要不要處理是你的決定。**同日追加**：你接著說明目標是要能用 `https://raspberrypi.tail8767da.ts.net/AI-Module/` 這種網址存取（Tailscale Serve 的 tailnet HTTPS 網域，跟你另外裝的 DuckDNS 是兩個不相關的機制，`.ts.net` 結尾只會來自 Tailscale）。查證 Tailscale 官方文件（`tailscale.com/kb/1242/tailscale-serve`、`.../1153/enabling-https`）確認：(a) `tailscale serve --set-path=/AI-Module <target>` 是把完整路徑（含 `/AI-Module` 前綴）原樣轉給後端，不會幫忙 strip 掉前綴；(b) 要先在 admin console 的 DNS 頁面開 MagicDNS + HTTPS Certificates，每台機器再各自跑一次 `tailscale cert` 才能簽出 Let's Encrypt 憑證。據此把上面提到的既有 bug 一起修掉、順便解決子路徑掛載問題：`web/backend/app.py` 的 `FRONTEND_DIR` 改指向真正的 `frontend/src/components/`、`PUBLIC_FILES` 換成 `{"index.html","style.css","action.py"}`（`script.js` 從沒真的存在過，一併從白名單移除）；新增 `MOUNT_PREFIX` 環境變數（預設空字串＝掛根目錄，行為跟改之前完全一樣），所有路由改掛在 `APIRouter(prefix=MOUNT_PREFIX)` 上，`MOUNT_PREFIX` 有設定時額外在根目錄加一個 307 轉址方便用 Pi5 區網 IP 直接測。`action.py` 對應改成不寫死 `/api/...` 這種帶開頭斜線的絕對路徑，一律用相對路徑（`"api/health"`）讓瀏覽器依目前頁面網址自動解析，`DETECT_WS_URL` 改用 `js.URL` 把 `"ws/detect"` 相對於 `window.location.href` 解析成絕對網址再換算 ws/wss，兩邊都不用寫死 `/AI-Module` 這個字串，同一份程式碼可以同時支援「掛根目錄」（本機、Render.com）跟「掛子路徑」（Tailscale Serve）兩種部署。用 `pytest`／`TestClient` 驗證過：不設 `MOUNT_PREFIX`（預設）時全部 10 項測試只剩 0 項失敗（原本失敗的 `test_public_assets_served` 也一併修好，改斷言 `style.css`/`action.py` 而不是從沒存在過的 `script.js`）；手動設 `MOUNT_PREFIX=/AI-Module` 跑起來後，`GET /AI-Module/api/health`、`GET /AI-Module/`、`GET /AI-Module/action.py` 都回 200，`GET /` 回 307 轉址到 `/AI-Module/`，確認前綴機制本身有正確運作。**尚未驗證的部分**：Tailscale 官方文件沒有明講 `--set-path` 是否會 strip 前綴，只是「不像有 strip」這個推論查不到反例；且這裡沒有實體 Pi5、也沒有你 tailnet 的 admin console 存取權限，沒辦法實際跑 `tailscale serve` 驗證這個假設，也沒辦法驗證 PyScript 前端在真瀏覽器透過這個網址開相機、串 WebSocket 是否真的正常——你依照下面附的 SSH 指令清單在 Pi5 上跑過、實際打開 `https://raspberrypi.tail8767da.ts.net/AI-Module/` 開相機測試沒問題後，記得回報讓我把這條寫進「已修正」；如果打開後 API 全部 404，最可能的原因就是我對「不 strip 前綴」的假設猜反了，把 `MOUNT_PREFIX` 環境變數整個拿掉（改回空字串、掛根目錄）應該就能解決，屆時再告訴我。

19. `web/frontend/src/components/action.py` 第 534 行 `from pyodide.ffi import create_once_callable`（見 to-do GPU 資訊卡任務中你順手回報的 IDE 錯誤，連帶問到第17條提過的 Pi5 部署現況）— Pylance/pyright 對這行報 `Import "pyodide.ffi" could not be resolved (reportMissingImports)`。根因跟第17條記錄的 `document`/`window`/`js` unknown import 是同一類問題（`pyodide`/`pyscript`/`js` 都是瀏覽器執行期 Pyodide 才注入的虛擬模組，本機/CI 裝不了），但當初只補了 `typings/js.pyi`、`typings/pyscript.pyi` 兩個存根，沒有涵蓋這行用到的 `pyodide.ffi`，所以單獨這一行仍然沒被消掉。**與這次改動無關但一併確認**：第17條記錄「Tailscale Serve 部署到 Pi5」這部分**當時就已標記「尚未完全驗證」**，是你要在實體 Pi5 上跑過 SSH 指令清單、實際開瀏覽器連 `https://raspberrypi.tail8767da.ts.net/AI-Module/` 測試相機/WebSocket 後回報——查了一遍 ErrorLog 全文跟 git log，找不到後續回報「Pi5 驗證通過」的記錄，所以目前狀態是「程式碼面已經為 Pi5 子路徑部署做好準備（`MOUNT_PREFIX`／相對路徑 API／`js.URL` 換算 ws 網址），但沒有證據顯示已經實際在 Pi5 上跑起來過」，不是「已部署完成」。**2026-08-24 已修正**（僅這行 IDE 警告）：仿照 `typings/js.pyi`／`typings/pyscript.pyi` 的寫法，新增 `typings/pyodide/__init__.pyi`、`typings/pyodide/ffi.pyi`（都是 module-level `__getattr__(name) -> Any`），不影響瀏覽器實際執行（瀏覽器讀的是 Pyodide 真的注入的 `pyodide.ffi`，不是這兩支存根檔）。用 `pyright web/frontend/src/components/action.py` 驗證：修前 `1 error`（就是這行），修後 `0 errors, 0 warnings, 0 informations`；全文搜尋確認專案裡只有這一處用到 `pyodide.ffi`。

18. `tranning/kernel_method.py`（新增的 Kernel Method 骨架，比照 `RNN.py`/`CNN.py` 風格，用 `make_moons` 對照 linear kernel vs rbf kernel SVM）— 執行時在 `from sklearn.datasets import make_moons` 就 `ModuleNotFoundError: No module named 'sklearn'`。查證後發現這其實是第 1 條「已修正」的舊坑重新出現：`pip show scikit-learn joblib` 確認目前 `.venv` 裡這兩個套件都沒裝（`numpy` 有裝），但第 1 條的修正對象是搬遷前的 `frontend/src/components/machine_learning.py`，當時裝套件的環境跟現在 `tranning/` 用的 `.venv` 不一定是同一份（專案歷經 `frontend/src/components/` → `app/components/` → `lib/components/`／`tranning/` 好幾次搬遷改名，`.venv` 中途疑似重建過）；也就是說 `tranning/machine_learning.py`（沿用同一支檔案搬過來的）在目前這份 `.venv` 底下其實也一樣會噴同樣的 `ModuleNotFoundError`，不是這次新增 `kernel_method.py` 才產生的新問題，只是這次先撞到。**2026-08-23 已修正**：`pip install scikit-learn joblib`（連帶裝入相依的 `scipy`/`threadpoolctl`/`narwhals`）。用 `python tranning/kernel_method.py` 實際執行驗證：linear kernel accuracy 0.8667、rbf kernel accuracy 0.9000（核方法在半月形非線性資料上確實優於線性核，驗證了核技巧的效果），模型存檔/讀檔/預測都正常無例外。

20. `tranning/tools.py` 第 94 行 `_SEARCH_PATTERNS`（見 to-do #15 同一批「知識詢問句型」規則）— 你回報問「你知道三星最新款的手機與手錶?」（西式問號結尾，沒有「嗎」），完全沒有比對到搜尋句型，整句掉回死記式 GRU 聊天模型（`chat_runs` checkpoint，見 to-do #12 記錄的天花板），硬拼出「我是 AI, PI的 P P P AI P AI」這種不成句的亂碼——不是「沒有上網能力」這件事本身的問題（那是 Rule 06 刻意設計），而是本來就該走 `_lookup()` 網路查詢路由的句子，卻被漏接掉回最不適合的死記模型。根因：`^你(?:認識|知道)\s*(.+?)\s*嗎[!?？]*$` 這條規則把「嗎」寫成必要字元（不是 `(?:嗎)?` 這種可省略寫法），只吃「你知道 X 嗎」，吃不到「你知道 X?」這種只用問號、口語上省略「嗎」的問法——跟 to-do #15 修的「開頭語氣詞前綴」是同一批規則、不同位置的同類疏漏（那次漏開頭，這次漏句尾）。**2026-08-25 已修正**：把該行改成 `^你(?:認識|知道)\s*(.+?)\s*(?:嗎)?[!?？]*$`，「嗎」變成可省略，兩種問法都能命中同一條規則、指到同一個 `_lookup()`。新增回歸測試 `test_route_reply_search_pattern_fires_without_trailing_ma`（`tranning/test_tools.py`），monkeypatch `_lookup()` 直接驗證這句話會被正確路由並帶出被抓到的主題字串；`pytest tranning/test_tools.py` 48 項全過（原本 47 項都沒被改壞）。另外用 `chats.smart_reply_traced()` 端對端重跑一次原句，確認不再落到 GRU 聊天模型，而是正確顯示「比對到搜尋句型，查詢主題「三星最新款的手機與手錶」」並嘗試真的查網路（這台機器這次查詢沒有查到結果，屬於資料源本身查無資料，不是路由邏輯的問題）。

21. `tranning/MentalHealth/test_action.py` 第 10 行 — 這次要幫 `action.py`（`data/mental.csv` 情緒傾向分數模型）的 `predict_emotion()` 加 MC Dropout 信心度，先跑既有測試確認沒改壞東西時發現：`from web.admin.frontend.src.components.action import ...` 這行 import 路徑是錯的，跟 `tranning/MentalHealth/action.py` 完全對不上——`web/admin/frontend/src/components/action.py` 是另一支同名但完全無關的檔案（PyScript 寫的網頁前端元件，處理 fetch/Google 登入，沒有 `SCORE_MAX`/`predict_emotion`/`train` 這些名字），研判是複製別的測試檔案時路徑沒改對。這個錯誤匯入讓整份 `test_action.py` 在 pytest collect 階段就 `ModuleNotFoundError`，兩項測試從頭到尾沒被執行過，`action.py`（含這次要接進 `chats.py` trace 的訓練成果）等於完全沒有自動化測試在保護。同時發現 `.venv` 沒裝 `scikit-learn`（`action.py` 模組層級 `from sklearn.model_selection import train_test_split`），是第 1／8／10／18 條同一類「環境缺套件」問題的重演，這次連單純呼叫 `predict_emotion()` 做推論（不需要真的訓練）都會被這行卡住。**2026-09-10 已修正**：import 改成 `from action import ...`；`.venv` 補裝 `scikit-learn`（連帶裝入 `scipy`/`joblib`/`threadpoolctl`）。`pytest tranning/MentalHealth/test_action.py` 2 項全過；另外用既有的 `tranning/MentalHealth/emotion_runs/` checkpoint（2026-09-02 訓練，`generalization_report.json` 顯示 64 筆held-out 測試 MAE=0.321、四捨五入命中率 79.7%）實際跑 `predict_emotion()` 驗證推論路徑本身也正常。

22. `tranning/transformer_chat.py`（見 [[project_transformer_emotion_retrain]] 2026-09-10 已上線的 Transformer 聊天模型）— 你回報對 sinco 輸入「我今天心情不好」，回覆「太好了，是什麼開心的事呢？每一刻都在想你說出這麼大的事。」，完全文不對題（把負面情緒當成正面情緒回應）。查 `data/pairs.json` 確認訓練資料本身沒問題：「我心情不好」→「怎麼了,想聊聊嗎」、「我今天心情不好，想要你抱抱。」→「來，過來，我抱你，什麼都先別想，有我在。」都是正確標註，不是資料標錯。實際對同一句「我今天心情不好」連續取樣 8 次重現：3/8（37.5%）開頭直接接到「太好了，是什麼開心的事」這個對應「我很開心」/「我今天心情很好」的正面模板（負面情緒與正面情緒兩個 prompt 只差「不」一個字，模型把「不好」的否定詞忽略掉了），其餘幾句雖然沒有極性錯誤但陸續出現語意不連貫/文法破碎（例如「好，是我每個人的都在想什麼時候能趕忙碌你找我聊聊」）；反過來測「我今天心情很好」也观察到同一種漂移（「太好了，是什麼開心的事呢。那就是想把哪個感覺放在懷煩的人心上，這份壓力著在大自然下...」）。**根因**：這是 [[project_transformer_emotion_retrain]] 2026-09-10 update #3 已經記錄在案、承認過的容量天花板——17M 參數規模的 from-scratch Transformer 對「不好」這種否定詞的區辨能力不夠穩定，容易跟形似的正面句型混淆，且較長的自由生成後段本來就會漂移不連貫，不是新問題，是既有已知限制第一次被實測量化出具體失敗率（37.5%）。**尚未修正**：不是資料錯誤、也不是簡單的 decoding 參數問題（那一輪已經在 update #3 修過重複迴圈），真的要動的話是「加更多含否定詞對比的資料再微調」或「換更大的模型」這兩個選項，兩者都涉及 GPU 訓練時間，依 [[feedback_no_unattended_long_training]] 先問過你的意向再動手，這次先只記錄現況。

23. `tranning/schmitt_trigger_train.py`（新功能：擴充模型讓 PCB 設計能力涵蓋史密特觸發電路，你要求「跑五次訓練，訓練出錯誤要總結出錯誤點在哪並修正，第五次還是錯誤就要輸出到 ErrorLog 當中」，見 to_do_list.md #29）— 手上完全沒有史密特觸發電路的真實訓練資料，先寫 `tranning/data/schmitt_trigger_gen.py` 用座標運算（不是手打 S-expression）產生 5 種電阻值變化的運算放大器版 + 1 種 74HC14 邏輯閘版共 6 種唯一電路、18 筆中文設計需求↔KiCad S-expression 問答對，每筆生成後直接餵給既有的 `tranning/circuit_rule_check.py`（to-do #23 的從零 ERC）驗證零 error 才收錄，資料本身沒有結構性錯誤。訓練骨架重用 `transformer_chat.py` 既有的 GPT/pretrain/finetune（零新模型程式碼），跑了 5 次真實訓練，每次都是真的錯誤+真的修正，不是重複同一件事：第1次（block_size=2048、reply=完整檔案含 lib_symbols、60/80 epoch）train_loss 停在 0.87、val loss 都還在下降就結束，生成結果括號數對不上（208 開/257 閉），明顯是文件還沒學會怎麼收尾；第2次想拉長 epoch 數重跑時發現 `main()` 的 argparse 預設值（60/80）忘記跟著 `train()` 函式本身的新預設值（150/400）一起改，一個很單純的程式碼疏漏，導致第2次其實只用舊的低 epoch 數又跑了一次，已修正 CLI 預設值；第3次（修正後真的跑 150/400，實際 early stop 在 74/189）train_loss 降到 0.50，但就算改用近乎貪婪解碼（temperature=0.05, top_k=1）+ 拉高 repetition_penalty=1.8 做診斷探測，生成結果從一開始的幾個 token 順序就是錯的，確認是真的欠擬合（underfitting），不是取樣隨機性的問題——根因判斷是 reply 目標文字裡 lib_symbols 樣板（每筆幾乎逐字元相同）佔了近半篇幅（約 1700 BPE token 中的一半），模型的容量/訓練訊號被拿去死背這段從不變化的樣板；第4次把 SFT 目標改成只生成「本體」（放置的元件/導線/標籤，lib_symbols 改成 `wrap_kicad()` 事後用程式接回去，不用模型生成），目標長度砍半（約 750 token），順勢把模型加大（n_layer 4→6, n_embd 128→192, block_size 2048→1024），結果反而更差：18 筆裡隨機切 15% 驗證集，剛好把少數類別（3 筆 74HC14 版本）不成比例地切進驗證集，val_loss 幾乎立刻打平在 1.30（比第3次的 0.66 還差），patience=40 正確地提早停止，但生成結果仍然無法解析（`list index out of range`）；第5次改成 `finetune(..., val_data_path=PAIRS_PATH)`，讓全部 18 筆同時當訓練集與驗證集（沿用 `transformer_chat.pretrain()` 本來就有的「資料太小時拿訓練集當驗證集」慣例，因為這個資料規模下「近乎全部背起來」本來就是目標，不是要考泛化），這次終於收斂良好：train_loss 0.16、val_loss 0.10（epoch 156 提早停止）。**但生成結果第五次仍然失敗**：對訓練集內的提示「幫我設計一個史密特觸發電路，R1 用 10k，R2 用 100k」，本體文字 70 個開括號只對到 69 個閉括號——實際比對發現 R1 的 `(symbol ...)` 區塊少了一個收尾的 `)`，直接接到下一個 OpAmp 的 `(symbol ...)`，且 OpAmp 那個元件的 Reference 屬性被錯誤填成鄰近區塊的 "R2"（應為 "U1"）；換成完全貪婪解碼（temperature≈0, top_k=1）在同一份 checkpoint 上重測，括號數不對稱的狀況還略微更差（71 開/69 閉），排除是取樣參數的問題。**尚未修正**：18 筆資料、6 種唯一電路本體（每份約 700-900 BPE token、深度巢狀括號結構）對這顆從零訓練、6 層/192 維的小型 Transformer 來說，train_loss 卡在 0.16 這個水準，還沒到 `chats.py` GRU 在 28 筆資料上能做到的 0.0001 那種近乎精確背誦的程度，多巢狀括號要做到「完全平衡」看起來需要更多/更多樣的訓練範例，或是更大的模型／更長的訓練時間，這已經超出本次「5 次訓練」授權範圍，依 [[feedback_no_unattended_long_training]] 不在同一份資料/架構上繼續重跑第 6 次，先誠實記錄現況，是否要投入更多資料標註或 GPU 時間由你決定。產物：`tranning/data/schmitt_trigger_gen.py`（產生器+ERC 自我驗證）、`tranning/schmitt_trigger_train.py`（train/design 指令）、`tranning/test_schmitt_trigger_train.py`（5 項測試，`pytest tranning/test_schmitt_trigger_train.py tranning/test_circuit_rule_check.py tranning/test_kicad_dataset_convert.py tranning/test_transformer_chat.py` 27 項全過，涵蓋資料產生器/ERC 整合，不含大規模真實訓練，那部分靠這次手動 5 次真實跑驗證）、`data/schmitt_trigger_pairs.json`、`tranning/data/schmitt_trigger_corpus.txt`、checkpoint 在 `tranning/schmitt_pretrain_runs/`／`tranning/schmitt_chat_runs/`。

24. `tranning/calculus_solver.py` — 你問 sinco「x^2+1, x 微分後是多少?」，回覆的是一般聊天模型的答案，完全沒進到微積分邏輯。查 `parse_and_solve()` 的 `_DERIVATIVE_PATTERNS`：四個既有 pattern 都是 `$` 錨定固定句型（「XXX 的微分／導數」「微分: XXX」「derivative of XXX」「d/dx(XXX)」），你這句「XXX, x 微分後是多少?」既不是「的微分」結尾，句尾又多了問號，四個 pattern 全部沒對上，`parse_and_solve()` 依文件說明的設計（看不懂就回 None，不用猜的）回傳 None，`tools.py` 的 `route_reply()` 因此判定不是微積分請求，掉回一般聊天模型接手，但聊天模型完全沒學過微積分，才會答不出來——不是模型算錯，是句型沒被規則認出來，根本没進到 sympy 那一步。**2026-09-11 已修正**：(a) `parse_and_solve()` 一開始先 `text.rstrip("?？!！。")` 去掉句尾標點，四個既有 pattern 不用重寫也能吃到「XXX 的微分？」這類問句；(b) `_DERIVATIVE_PATTERNS` 新增一條 `^(.+?)\s*(?:[,，]\s*(?:對\s*)?x\s*)?微分後(?:是多少|為何|等於多少)?$`，涵蓋口語問法「EXPR[, x] 微分後是多少」。驗證：`tranning/test_calculus_solver.py` 新增 `test_derivative_spoken_suffix_with_trailing_question_mark` 直接餵你原句，斷言 `sympy` 微分結果一致；`pytest tranning/test_calculus_solver.py` 18 項全過；CLI 手動跑 `python tranning/calculus_solver.py "x^2+1, x 微分後是多少?"` 正確回傳 f'(x) = 2x 的完整解題過程。

25. `tranning/calculus_generator.py`（新功能，非錯誤修正——你要求微積分模型「更進階」，見
    `C:\Users\roy\.claude\plans\spicy-swimming-sunset.md` Phase 1）— 原本
    `_explain_term_derivative()`/`_explain_term_antideriv()` 只認 3 種「純項」
    (`is_polynomial`、`c*sin(kx)`/`c*cos(kx)`、`c*exp(kx)`，inner 必須是線性 `k*x`)，
    乘積(`x*sin(x)`)、商(`x/(x+1)`)、非線性合成(`sin(x^2)`)一律 fallback 成無規則
    名稱的「直接微分（sympy 計算）」句子——答案一直是對的(sp.diff/sp.integrate 本身
    能處理任意初等函數)，只是詳解步驟不像人教的。**已擴充**：新增
    `_fraction_parts_if_quotient()`(用 `sp.fraction()` 抓真正 x 相關的分母，判斷順序
    在乘積律之前，避免 `x/(x+1)` 的內部 Mul 表示法(`x, (x+1)**-1`)被誤判成乘積)、
    `_product_parts_if_product_rule()`(Mul 且恰好 2 個含 x 因子)、`_match_pure_outer()`
    (推廣舊的 `_is_pure_trig`/`_is_pure_exp`，inner 不再限制線性——線性 inner 沿用
    舊有措辭不變，非線性 inner 才用新的「令 u=...」鏈鎖律措辭，向後相容)三個結構判斷
    式，`_explain_term_derivative()` 判斷順序：冪法則→商法則→乘積律→鏈鎖律(含推廣)→
    fallback。積分側新增 `_match_u_substitution()` 辨識「係數×inner'(x)×f(inner(x))」
    的湊微分(u代換)形式(例如 `x*sin(x^2)`)，能辨識時給「令 u=...」步驟，辨識不到時
    (例如 `x*sin(x)` 需要分部積分，這模組不處理)維持原本 fallback，不硬套規則。quiz
    出題端新增 `_term_product`/`_term_chain`(微分)、`_term_u_sub_for_integral`(積分)
    三個 term builder 加進抽樣池，讓「出一題微積分」也會出到這些新題型，不是只有
    使用者自己打合成算式才吃得到。過程中發現一個副作用：`_term_chain`/
    `_term_u_sub_for_integral` 一開始讓 `exp` 的 inner 也能加常數(例如 `exp(x^2+2)`)，
    但 `explain_derivative()`/`explain_integral()` 開頭的 `sp.expand()` 預設會把
    `exp(a+b)` 拆成 `exp(a)*exp(b)`，導致印出「3exp(2)exp(x^2)」這種多一個常數指數
    因子的怪異格式(數學上沒錯，但不像課本寫法)——`sin`/`cos` 沒有這個 expand 行為，
    已修正成只有 `sin`/`cos` 的 inner 才加常數、`exp` 的 inner 維持純 `x^n`。驗證：
    新增/改寫 `test_calculus_generator.py`(乘積律/商法則/非線性鏈鎖律/線性鏈鎖律措辭
    不變/三因子以上 fallback/湊微分辨識成功與正確拒絕`x*sin(x)`/quiz builder 有出現
    新規則共 9 項)、`test_calculus_solver.py`(自由輸入 `"x*sin(x) 的微分"`、
    `"sin(x^2) 的微分"` 2 項)，`pytest tranning/test_calculus_generator.py
    tranning/test_calculus_solver.py tranning/test_tools.py` 108 項全過；另外寫一段
    腳本跑 300 個 seed 的 `generate_problem(topic="derivative")` 掃過 `exp(`後面接數字
    這種格式，確認 0 筆殘留。CLI 手動跑 `x*sin(x) 的微分`／`x/(x+1) 的微分`／
    `sin(x^2) 的微分`／`x*sin(x^2) 的積分` 詳解格式正常可讀。**尚未做的路線圖**(見
    plan 檔 Phase 2-4，需要你逐階段確認才會做)：更多函數類型(log/sqrt/反三角/雙曲)、
    高階微分與泰勒展開、多變數微積分(偏微分/梯度)——多變數那塊要新增第二個符號，
    改動面最大，規劃獨立一輪 session 處理。

26. `tranning/calculus_generator.py` + `tranning/calculus_solver.py`（新功能，非錯誤修正
    ——接續第 25 條 Phase 1，你確認繼續做 Phase 2：更多函數類型，見
    `C:\Users\roy\.claude\plans\spicy-swimming-sunset.md` 路線圖）— 原本
    `calculus_solver.py` 只能解析 sin/cos/tan/exp/ln/log/sqrt，`calculus_generator.py`
    的「純項」辨識(`_match_pure_outer`，Phase 1 新增)也只認 sin/cos/exp 三種，log/sqrt
    雖然能解析但沒有課本公式步驟，一律 fallback 成無規則名稱的「直接微分（sympy 計算）」；
    反三角(arcsin/arccos/arctan)、雙曲(sinh/cosh/tanh)完全不能解析。**已擴充**：(a)
    `calculus_solver.py` 的 `_LOCAL_DICT` 加入 `asin`/`arcsin`、`acos`/`arccos`、
    `atan`/`arctan`(兩種拼法都接受)、`sinh`/`cosh`/`tanh`；(b) `calculus_generator.py`
    的 `_match_pure_outer()` 辨識函數清單從 3 個(sin/cos/exp)擴充到 11 個(新增
    log/sqrt/asin/acos/atan/sinh/cosh/tanh)，`_explain_term_derivative()` 新增
    `_DERIVATIVE_FORMULA_LABELS` 對照表，給這 8 個新函數線性 inner 時對應的課本公式
    措辭(例如 `(ln x)' = 1/x`)，非線性 inner 沿用 Phase 1 已建好的通用「令 u=...」鏈鎖律
    措辭(不用重寫，`_match_pure_outer` 本來就是通用的)；sin/cos/exp 原本的措辭完全沒動
    (用 `elif name == "exp"` 保留舊分支，只有新函數才走新的 dict 對照分支)。(c) 積分側
    只挑 3 個「反導函數是乾淨一行公式」的新函數(sqrt/sinh/cosh)加 `_ANTIDERIV_FORMULA_LABELS`
    直接公式措辭，並擴充 `_match_u_substitution()` 的可辨識函數清單納入這 3 個，讓
    `x*sqrt(x^2+1)` 這類湊微分也認得；log/arcsin/arccos/arctan/tanh 的反導函數需要分部
    積分(例如 `∫ln(x)dx = x·ln(x)-x`)，不是乾淨公式，刻意不勉強套用規則名稱，維持
    fallback(答案仍正確，只是沒有規則名稱)。(d) `_fmt()` 新增 sympy 函數名稱→中文課本
    慣用寫法的顯示轉換(`log`→`ln`、`asin`→`arcsin`、`acos`→`arccos`、`atan`→`arctan`、
    `sqrt`→`√`)，過程中發現一個一致性 bug 順手修掉：鏈鎖律 fallback 措辭與湊微分措辭
    原本直接把 `_match_pure_outer()`/`_match_u_substitution()` 回傳的原始比對名稱(例如
    `"log"`)塞進句子，導致同一行內算式本身顯示成 `ln(x^2+1)`、規則名稱卻印成
    `log(u)`，兩種寫法混用——新增 `_DISPLAY_NAME` 對照表統一這兩處的顯示名稱。(e) quiz
    出題端新增 `_term_more_functions`(微分)、`_term_more_functions_for_integral`(積分
    直接公式)兩個 term builder，並把 `_term_u_sub_for_integral` 的候選函數也從
    sin/cos/exp 擴充到含 sqrt/sinh/cosh。驗證：新增/擴充
    `tranning/test_calculus_generator.py`(8 個新函數線性 inner 公式措辭+正確性的
    parametrize 測試、非線性 inner 鏈鎖律措辭含正確顯示名稱、`_fmt` 顯示名稱轉換、
    sqrt/sinh/cosh 積分公式、sqrt 湊微分、log/asin 積分刻意 fallback、quiz builder 兩項
    smoke test 共 9 項新測試)、`tranning/test_calculus_solver.py`(8 個新函數自由輸入
    parametrize 測試 + sqrt 積分共 2 項)，`pytest tranning/test_calculus_generator.py
    tranning/test_calculus_solver.py tranning/test_tools.py` 133 項全過；另外寫腳本跑
    derivative/integral 各 400 個 seed，逐一用 `sp.diff`/`sp.integrate` 重新驗算答案，
    0 筆不一致。CLI 手動跑 `log(x)`/`sqrt(x)`/`arcsin(x)`/`sinh(2x)` 的微分、
    `sqrt(x)`/`x*sqrt(x^2+1)` 的積分，詳解格式正常、顯示名稱(ln/√/arcsin)一致。**尚未做
    的路線圖**(見 plan 檔 Phase 3-4，需你逐階段確認才會做)：高階微分與泰勒展開、多變數
    微積分(偏微分/梯度，需要新增第二個符號，改動面最大)。

27. `tranning/calculus_generator.py` + `tranning/calculus_solver.py`（新功能，非錯誤修正
    ——接續第 25、26 條，你確認繼續做 Phase 3：高階微分 + 泰勒/馬克勞林展開，見
    `C:\Users\roy\.claude\plans\spicy-swimming-sunset.md` 路線圖）— 原本完全沒有二階
    以上導數、也沒有級數展開的功能。**新增**：(a) `calculus_generator.py` 新增
    `explain_nth_derivative(expr, n)`：對 `explain_derivative()` 重複套用 n 次(每次把
    上一階的結果當輸入再算一次)，每一階完整保留 `explain_derivative()` 原本的逐項規則
    說明(前綴 `[第 k 階]`)，不是把每階壓縮成一行「再微分一次」——這樣 Phase 1/2 已經
    做好的乘積律/商法則/鏈鎖律/8 種新函數公式，在高階微分的每一階都自動可以用到，
    不用重寫規則邏輯。(b) 新增 `explain_taylor_series(expr, point, order)`：逐階計算
    `f^(k)(point)/k!` 係數(直接用 `sp.diff(expr, x, k).subs(x, point)`，不是呼叫
    `sp.series()`，因為這裡只需要單點的導數值而非完整符號導函數)，`point=0` 時
    `topic_zh` 顯示「馬克勞林級數」，其餘顯示「泰勒級數」。(c) `calculus_solver.py`
    新增中文序數解析(`_CN_DIGITS`，一~十)+ 4 組新句型：「EXPR 的[第]N階導數/微分」
    (中文序數或阿拉伯數字皆可)、`d^n/dx^n(EXPR)` 記法、「EXPR 在 x=a 展開到[第]n階
    [的]泰勒級數」、「EXPR 的[第]n階馬克勞林展開/級數」。驗證：新增
    `test_calculus_generator.py`(7 項：反覆微分正確性、step 數量隨階數增加、n<1 拒絕、
    泰勒展開對照 `sp.series()`、point=0 標記馬克勞林、負階數拒絕)、
    `test_calculus_solver.py`(5 項：中文序數/阿拉伯數字/d^n-dx^n 三種二三階導數句型、
    非零點泰勒展開、馬克勞林展開)，`pytest tranning/test_calculus_generator.py
    tranning/test_calculus_solver.py tranning/test_tools.py` 144 項全過；另外寫腳本對
    7 個函數各測 1-3 階導數、6 組泰勒/馬克勞林案例，逐一對照 `sp.diff`/`sp.series`
    驗算，全部一致。CLI 手動跑 `sin(x) 的三階導數`、`d^2/dx^2(x^3+2x)`、
    `exp(x) 在 x=0 展開到第4階泰勒級數`、`sin(x) 的5階馬克勞林展開`，輸出格式與數值
    正確。**刻意不做的範圍**：這兩個新功能只接在 `calculus_solver.py` 的自由輸入解題
    路徑，沒有接進 `generate_problem()`/`GENERATORS` 的「出一題微積分」隨機出題流程
    (也沒有對應加 `tools.py` 的關鍵字路由)——效益評估後認為使用者的核心需求是「能解
    使用者自己打的進階算式」，出題端的擴充範圍較大(需要新 quiz topic、新 term builder、
    新 tools.py 路由)且非本次確認的路線圖項目，先不做，之後若需要再另外確認。**尚未做
    的路線圖**(見 plan 檔 Phase 4，需你確認才會做)：多變數微積分(偏微分/梯度)，需要
    新增第二個符號，改動面最大，規劃獨立一輪 session 處理。

28. `tranning/calculus_generator.py` + `tranning/calculus_solver.py`（新功能，非錯誤修正
    ——接續第 25、26、27 條，你確認做完 Phase 4：多變數微積分(偏微分/梯度)，見
    `C:\Users\roy\.claude\plans\spicy-swimming-sunset.md` 路線圖，至此四階段全部完成）—
    原本整個模組只有單一符號 `x = sp.symbols("x")`，`calculus_solver.py` 的
    `_parse_expr_text()` 明確拒絕任何 `free_symbols` 不是 `{x}` 子集的算式，完全不支援
    偏微分/梯度。**新增**：(a) `calculus_generator.py` 新增模組級第二符號
    `y = sp.symbols("y")`，以及 `explain_partial_derivative(expr, var)`、
    `explain_gradient(expr)` 兩個函式。這兩個函式**刻意是獨立、自成一體的實作，沒有
    重構 Phase 1-3 既有的 `_explain_term_derivative()` 系列函式**(那些函式從頭到尾寫死
    綁定模組級 `x`，例如 `_WILD_C = sp.Wild("c", exclude=[x])`、`term.is_polynomial(x)`、
    `sp.diff(term, x)`——要讓它們改成支援任意變數，需要把 Phase 1-3 每一個規則判斷式都
    加一個 `var` 參數並全面重新驗證，風險與工作量都遠超這一階段範圍，plan 檔規劃階段就
    已經決定不這麼做)；新函式只借用 `sp.diff(expr, var)` 本身(sympy 本來就會把「不是
    要微分的那個符號」自動視為常數，不需要額外邏輯)，配上自己寫的、比 Phase 1-3 簡單
    的說明文字(「將 {other} 視為常數」+ 逐項結果)。`_fmt()` 的「blanket `*` 移除對單一
    符號安全」這個既有假設在雙變數下依然成立：`x*y` 移除 `*` 變成 `xy`，這正是常見的
    並列相乘慣例(跟 `3x` 代表 `3*x` 是同一種慣例)，只有在模組真的會生出一個字面上叫
    `"xy"` 的符號時才會產生歧義，這個模組不會這樣做。(b) `calculus_solver.py` 新增獨立
    的 `_LOCAL_DICT_XY`/`_parse_expr_text_xy()`(允許 x 和/或 y，不動原本只認 x 的
    `_LOCAL_DICT`/`_parse_expr_text()`)，加 4 組新句型：「EXPR 對 x/y [的]偏微分」、
    `∂/∂x(EXPR)`/`∂/∂y(EXPR)` 記法(刻意只認 `∂` 不認 `d`，避免跟既有
    `d/dx(EXPR)` 單變數記法衝突而誤路由)、「EXPR 的梯度」、`gradient of EXPR`。驗證：
    新增 `test_calculus_generator.py`(6 項：偏微分數值正確性、常數措辭正確性、拒絕非
    x/y 變數、拒絕額外自由符號、梯度正確性、純 x 表達式的退化梯度)、
    `test_calculus_solver.py`(4 項：中文句型、∂記法、**`d/dx` 仍正確走原本單變數路徑
    的回歸防護測試**、梯度句型)，`pytest tranning/test_calculus_generator.py
    tranning/test_calculus_solver.py tranning/test_tools.py
    tranning/test_logic_reasoning_generator.py tranning/test_word_problem_generator.py`
    373 項全過(含另外兩支重用 `calculus_generator.format_problem()`/`format_question()`
    的模組，確認新增的 `y` 符號沒有連帶影響)；另外寫腳本對 8 組雙變數算式各驗證
    ∂/∂x、∂/∂y、梯度，全部對照 `sp.diff` 一致。CLI 手動跑「x^2\*y+y^3 對 x 的偏微分」、
    `∂/∂y(x^2*y)`、「x^2\*y+y^3 的梯度」、`d/dx(sin(x))`(確認仍回傳
    `topic=derivative` 而非誤入新的 `partial_derivative` 路徑)，全部正確。至此微積分
    模組四階段路線圖(合成函數/更多函數類型/高階微分+泰勒/多變數)全部完成，plan 檔
    可視為結案。

29. `tranning/data/schmitt_trigger_gen.py` + `tranning/schmitt_trigger_train.py`（延續第 23 條，
    你要求「模型還要會史密特電路等 OPA 多項電子電路知識，上網搜尋，訓練可到五小時，錯誤修到
    第五次才寫 ErrorLog」，見 to_do_list.md #29 2026-09-11/12 追加記錄，這裡補完整的錯誤/修正
    脈絡）— 第 23 條記錄的 5 次舊嘗試最後卡在「18 筆/6 種電路，train_loss 只能到 0.16，生成
    結果括號數對不上」。這次先上網查證（見來源）反相/非反相放大器增益公式與比較器接法，
    `schmitt_trigger_gen.py` 新增 3 種拓樸（`build_inverting_amp`/`build_noninverting_amp`/
    `build_comparator`），語料擴到 54 筆/18 種電路，全部先過 `circuit_rule_check.py` 零 error
    才收錄。`.venv` 這次確認是 GPU 版 `torch 2.13.0+cu126`（`torch.cuda.is_available()=True`），
    訓練速度比舊的 CPU 環境快非常多。**錯誤 1**（第1次訓練後稽核發現）：「反相放大器」/
    「非反相放大器」提示詞只差一個「非」字，模型常把兩種拓樸的接線搞混（近乎貪婪解碼下 100%
    重現，非取樣隨機性）。**已修正**：非反相放大器提示詞改用更標準的「同相放大器」（查證過
    確實是常見中文電子學教材用詞），從根源避開這組最小差異字對。**錯誤 2**（同一次稽核發現）：
    74HC14 史密特反相器被生成成 `lib_id "Sinco:R"`（座標剛好跟正確元件重疊，`circuit_rule_check`
    抓不到，但元件類型是錯的）。第2次重訓後錯誤1、2 都消失（18/18 電路 ERC 全過、74HC14 那個
    也逐字元比對成功），但逐字元比對 18 種電路才發現**錯誤 3、也是最根本的問題**：電路「結構」
    18/18 正確，但電路「數值」（使用者要求的 R1/R2 阻值）**只有 1/18 是對的**——模型會生成一個
    看起來合理但答非所問的阻值組合，根因是這顆 6 層/192 維的小模型沒學會「把提示詞裡的數字原樣
    複製到電路裡」，記的是一個籠統但錯誤的數值關聯，不是真的在做複製。**已修正**（架構性修法，
    非調參）：比照 `wrap_kicad()` 把不變的 lib_symbols 交給程式而非模型的同一個思路，
    `build_corpus()` 讓每個電路的 R1/R2 一律用字面佔位符（`PLACEHOLDER_R1`/`PLACEHOLDER_R2`），
    訓練目標從「記住 18 種帶數值的電路」簡化成「記住 5 種帶佔位符的樣板」；新增
    `schmitt_trigger_train.fill_values()`，生成完後用正規表示式直接從使用者原始提示詞解析出
    R1/R2 代入，不再信任模型生數字。第3次重訓後端對端稽核全部 54 筆訓練提示詞：**54/54 結構
    通過 ERC、54/54 數值完全正確**，額外用 4 句訓練集以外的提示詞（全新數值組合、全新措辭）
    壓力測試：3/4 正確（含完全沒看過的 33k/330k、1k/10k 數值組合，證明是樣板+代入機制而非
    硬背），1/4 因為「R1 是 10k」這種未收錄的助詞使 `fill_values()` 的正規表示式沒抓到值，已
    順手擴充正規表示式涵蓋「是/為/設為」等常見說法。**錯誤 4**：接著新增第 6 種拓樸
    `build_voltage_follower()`（電壓隨耦器，查證中文教材確認用詞），第4次重訓後結構正確率從
    100% 退步到 91%（52/57），數值代入仍是滿分（57/57，證明架構性修法本身穩固，退步只發生在
    結構生成端）。查訓練曲線確認 val_loss 在 400 個 epoch 裡有 258 個卡在同一數值，是真的收斂
    到頂而非訓練不夠久（先查證再判斷，不用猜的）——判定是模型容量在加到第 6 種拓樸後到頂，
    這是這次自己主動加的範圍（不是你原始要求的必要項目），選擇**撤回**而非賭最後的訓練名額，
    退回驗證過 100% 正確的 5 拓樸版本。**錯誤 5**（第5次重訓，用完全相同的 5 拓樸設定重跑做
    確認）：結果沒有重現第3次的 100%，反而是 47/54（87%）ERC 通過（數值代入仍 54/54 滿分），
    7 個失敗全是 `floating_pin`、且集中在 R1=10k（全語料庫裡最常重複出現、最模糊的一個值）——
    用同一份 checkpoint 連續生成 3 次確認是這個 checkpoint 本身的問題、不是取樣隨機性。查
    `transformer_chat.py` 全文確認 `pretrain()`/`finetune()` 完全沒有設定任何 random seed，
    每次 `train()` 都是全新隨機初始化/洗牌，這足以解釋「相同資料、相同超參數，兩次訓練結果
    品質不同」——第3次是運氣好抽到一個好的初始化，第5次沒有。**累計已達你設定的 5 次錯誤
    上限，依指示停止訓練、寫入本條，不再進行第 6 次重訓**。**目前狀態老實記錄**：現在存在
    `tranning/schmitt_chat_runs/`／`tranning/schmitt_pretrain_runs/` 的 checkpoint 是第5次
    的結果（47/54 結構正確、54/54 數值正確），不是第3次那個曾經達到的 100%/100%——第3次的
    checkpoint 在第4、5次重訓時已被覆蓋，沒有另外備份。**架構本身已證明可行**（第3次真實達到
    過 54/54/54/54），真正欠缺的是訓練可重現性；下次要繼續的話，建議先在 `transformer_chat.py`
    的 `pretrain()`/`finetune()` 加對 `torch.manual_seed()`/`random.seed()` 的顯式控制，讓同一
    組資料/超參數的訓練結果可重現，才不會每次重訓都要看運氣，這是程式碼本身的缺口，不是資料
    或超參數的問題。**驗證**：`pytest tranning/test_schmitt_trigger_train.py
    tranning/test_circuit_rule_check.py tranning/test_transformer_chat.py
    tranning/test_kicad_dataset_convert.py` 29 項全過（含新增的 `fill_values()` 2 項測試）；
    產物：`tranning/data/schmitt_trigger_gen.py`（新增 `build_inverting_amp`/
    `build_noninverting_amp`/`build_comparator`/`build_voltage_follower`，後者未收錄進
    `build_corpus()`，函式保留供之後重新加入）、`tranning/schmitt_trigger_train.py`（新增
    `fill_values()`）、`data/schmitt_trigger_pairs.json`、`tranning/data/schmitt_trigger_corpus.txt`。
    來源：[circuitdigest.com 反相/非反相放大器增益公式](https://circuitdigest.com/tutorial/inverting-operational-amplifier-op-amp)、
    [electronics-tutorials.ws 比較器接法](https://www.electronics-tutorials.ws/opamp/op-amp-comparator.html)、
    [enroo.com 同相放大器教材](http://www.enroo.com/support/category1/dpjrmzs/76245016.html)、
    [opentech.com.tw 反相/非反相放大器及電壓隨耦器教學講義](https://www.opentech.com.tw/try/lcuaj2025512110108/zwo3wpz7uu2025512110108.pdf)。

## 修正日誌
1. `tranning/chats.py`（原 `frontend/src/components/chats.py`；你已把整個 components 資料夾獨立拉出來改名為 `tranning/`，之後聊天模型相關檔案都在這裡，`chat_runs/` checkpoint 也一起搬過去了）— 2026-07-25：把 `data/pairs.json` 從 4 筆佔位資料擴充到 28 筆，涵蓋身份自介、打招呼、道別、道謝、閒聊等類別，身份類回覆固定用英文原名「sinco」（未翻譯）。訓練驗證：`--epochs 50`（預設值）在這個資料量下明顯不夠，幾乎所有 prompt 收斂到同一句回覆（loss 卡在 2.7~3.8 沒降下去）；拉到 `--epochs 1500` 後 loss 降到 0.0011，對 10 句抽測（含「你是誰」「你是 ChatGPT 嗎」「hi」「謝謝」等）全部對到正確且不同的回覆，無互相混淆。**結論**：資料筆數增加時，需要的 epoch 數要跟著往上調，不是固定值；目前 28 筆規模下模型仍是死記式 Q&A（把每一句完整背下來），還不具備「類似問法舉一反三」的泛化能力，要有真正聊天感需要資料量遠大於 28 筆。
2. `tranning/transformer_chat.py`（見 to-do #12）— 兩個值得記住、以後可能還會遇到的通用結論：(a) 從零刻的深層 Transformer（decoder-only，多層殘差疊加）如果只用一般的 `nn.Linear` 預設初始化，未訓練模型的 loss 會遠高於理論基準線 `ln(vocab_size)`（實測 8 層、vocab_size=8000 時，一般初始化 loss=94.9，理論基準線只有 9.0），層數越深問題越明顯——判斷「這是初始化問題還是訓練資料/架構真的有問題」的快速檢驗法：拿一個全新建構、完全沒訓練過的模型跑一次 forward，loss 應該要落在 `ln(vocab_size)` 附近，如果差很多，先查初始化，不要急著懷疑資料或加大模型。修法是 GPT-2 論文的標準做法：每個殘差分支的輸出投影（attention 的 `proj`、MLP 最後一層）用 `std=0.02/sqrt(2*n_layer)` 縮小初始化。(b) `bpe_tokenizer.py` 的 BPE 訓練是教科書式的樸素實作（每次 merge 都重新掃過整個語料算 pair 頻率），複雜度約 `O(merge 次數 × 語料長度)`，22.5 萬字語料、vocab_size=8000（約 5000 次 merge）實測跑了 212 秒——這個量級可以接受，但如果之後語料量級再放大 10 倍以上，這支未優化的樸素版可能會變得太慢，屆時需要換成「只更新受影響的 pair 計數」的增量版寫法，而不是每次都整個語料重算。
