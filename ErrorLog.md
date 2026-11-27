# Error Log

## 2026-10-28：卷積／直方圖練習的公式與顯示紀錄

使用者要求for迴圈影像卷積、禁用NumPy，並確認均衡化採獨立Lenna練習。初次只擷取Word一般文字時Filter2空白，核對OMML後確認為1/25乘5×5全1，沒有猜核。來源512×512、灰階參考850×846，保留來源尺寸，不將灰階轉換或尺寸差異當卷積結果。Filter1保留原係數，Sobel保留負值；原始JSON與clipped/minmax顯示分離，避免8位元圖片掩蓋運算值。

均衡化依原練習Y/Cb/Cr公式與CDF-min查表，以純迴圈實作；常數圖不除以0。手算與實際圖結果通過20項測試及獨立審查，Y範圍34–224展開為0–255，尚待使用者驗收視覺效果。測試初版Pillow getdata出現棄用警示，已換getpixel；無產品計算錯誤或NumPy使用。來源只讀與合成輸入位元組保持有驗證；未取得實圖執行前SHA，故不宣稱實際前後hash一致。未修改CNN-MNIST、未執行訓練、未改原始圖片或練習文件。證據見team-run/image-processing/REVIEW.md、BUILD_CONVOLUTION.md、BUILD_HISTOGRAM.md。

## 2026-10-28：開放式電學首輪訓練完成但生成品質未達標

使用者核准後，v2資料以lr1e-5、batch2、22電學＋22重播、凍結embedding與前7層，實際完成3 epochs／66更新；最佳epoch3 train_loss3.7697、dev_loss5.4044。每epoch都有train/dev loss落差警示，保留不擅自調參。候選保存於tranning/gpt_electricity_open_runs/pilot_v1，正式基底未覆蓋或切換。

獨立核對14個tensor僅最後block與ln_f變動、全部finite。原始GPT同問題／seed／解碼比較：dev基底0/10、候選0/10；候選首次test0/10；已見train抽樣也0/5，100Ω20V及節點／網目回歸仍偏題。這輪並未達到解題目標，不能以loss下降視為成功。四筆一般chat逐字相同，但不足以保證全能力保留。全部原始回答先落地再讀鍵，未經公式工具，test已開封不得再當新盲測。保留候選、不晉升、不自行追加訓練；下一輪依#43另行確認。

訓後審查腳本初次read_text未指定UTF-8遇cp950解碼失敗，改encoding='utf-8'後完成，非訓練異常。證據見data/electricity_open/PILOT_V1_RESULT.md、output/electricity-open/pilot_v1_evaluation及team-run/electricity-open/PILOT_V1_REVIEW.md。此為未達標試驗紀錄，不寫已驗收成功日誌。

## 2026-10-28：基本電學開放題被回答成情緒陪伴

使用者回報「你會解基本電學嗎？」及100歐姆／20V求電流得到偏題文字。一般聊天仍呼叫正式 `gpt_chat_runs`，先前只做題庫準備，沒有更新權重；結構式輸出檢查無法保證語意相關。新增共用本機歐姆定律與明示固定電路的節點／網目教學，明確標示非模型生成；另備生成式監督微調候選資料與受控入口，並保留原始GPT盲答，沒有正式訓練或切換模型。

審查修正：初版regex的 `\\b?` 造成PatternError；初版忽略SI大小寫把MA當mA，並對交流題套直流公式，現已拒絕不支援前綴與交流／阻抗範圍。訓練準備初版以zip對應題目與鍵、未排除重播保留題、best_epoch僅文字，改為id配對、過濾與實際紀錄。數值初稿過於集中，保留草稿，正式候選v2改為多種電流與單位結果。v2生成時原始碼hash與當時首答一致；後續程式加固造成現碼hash不同，保留歷史指紋、不倒填，Reviewer已逐題重現42題與目標一致。

獨立驗證103項測試與1000組精確KCL／KVL／映射驗算通過；42題preflight無UNK、最長205 tokens。全22筆train首答已持久化，test未作答。742筆原聊天資料以六個情緒詞命中115筆僅為關鍵詞統計，不能推論偏題全由情緒資料比例造成。正式權重、原訓練程式與既有四選一草稿指紋未變。證據見team-run/electricity-open/BUILD_A.md、BUILD_B.md、REVIEW.md；待使用者實際操作及逐輪核准訓練，未寫入已驗收修正日誌。

探索指令另遇PowerShell內Get-Command找不到sinco（工具程序與使用者終端環境不同）、rg將Windows萬用字元路徑視為非法；未修改使用者profile或別名，未宣稱已驗證sinco入口指向。建議用專案Python直接啟動來源CLI來確認本輪程式。

一次合跑既有隨機測試將答案18與題幹180的子字串重疊誤判為洩漏；未改既有測試以遷就功能，後續獨立103項測試通過。此既有測試的隨機／子字串判定限制保留紀錄。

## 2026-10-27：基本電學先作答、再核對的資料準備

43檔只讀盤點後僅24題可可靠核對為初始無圖MCQ，106/108掃描卷、圖題／承上、OMML缺失、官方4送分及不可靠試算表公式均未冒充標準題。試算表電荷常數1.602e-22與同表1.602e-19矛盾，保留原檔隔離，未重算公式或執行附件Python。研究時小數誤切題號與一次不完整Unicode正則PatternError已停止；正式題號切分避開小數選項，使用完整原頁及官方表核對。一次查找不存在context模組的rg錯誤已改讀實際context_format.py，未動該檔。

早期審查指出correct_option空值／多字元、衝突重複只留第一筆、重排選項未同群及devfeedback風險，已在正式產檔前修正。資料v1為18train／3dev／3test；推論不讀keys，原始作答先fsync，再由獨立評分步驟讀鍵；列舉／矛盾不給分，非MCQ待人工，feedback限train且無正解。61項工程測試通過。

直接正式GPT首答train9＋dev3共12題，UNK0／無截斷，全12偏題且無合法末行，分數0/12；三筆一般聊天原文另存，test未用。這是現有模型能力缺口，不以工具代答或把格式修正當物理推理。既有GRPO子字串reward與全組零分無有效相對信號，不直接套用。6份受保護檔hash相同，沒有optimizer更新。證據見team-run/basic-electricity/BUILD_LOG.md與output/basic-electricity；待使用者確認下一輪，未寫已驗收成功日誌。

## 2026-10-26：圖片搜尋誤入聊天與中文網址百分比碼

最終獨立審查：英文圖片搜尋處理紀錄已改依輸入語言顯示；Reviewer重跑六組測試160項全部通過、零略過，先前Tk初始化略過未在此輪重現。詳見team-run/image-search/REVIEW.md。

使用者輸入「你尋找有關周柯宇的圖片」時，既有路由只支援「辨識圖片＋路徑／網址」，沒有圖片搜尋分支，因而落入舊聊天模型並回覆無關文字。新增共用網路圖片搜尋、現有 characters.pt 目標篩選與獨立圖片預覽；未知姓名、零匹配、缺模型或推論失敗的搜尋候選均標明未確認，沒有重訓。既有維基及搜尋來源網址另以可讀中文標籤顯示，點擊仍用原始網址，保留 %2F／%26 與簽名。

探測 Bing 頁面曾有圖片資料，後續只回 script；DuckDuckGo 驗證頁、Baidu antiFlag、Yahoo500均未繞過。改採實測成功的360公開圖片JSON優先、Bing備援；部分第三方圖片403會略過。第一次探測輸出遇 Windows cp950 UnicodeEncodeError，後續探測設定 PYTHONIOENCODING=utf-8；產品CLI原有UTF-8設定保留。一次讀取不存在的 image_renderer.py 索引路徑失敗，確認既有檔案實為 detection_overlay.py，未建立錯誤路徑或修改該檔。

新增Tk連結測試首次以錯誤的Text.tag_bind查詢簽名造成TypeError，已顯式傳None修正並重跑；反覆建立Tk root仍會偶發既有初始化skip。相關測試159 passed／1 skipped；兩項新增GUI渲染測試通過。真實網路共用路由31.7秒取得三張候選、YOLO目標匹配0張且明示未確認；真Tk三分頁預覽開關煙測通過。權重SHA前後相同，不將工程測試當人物身份準確率驗收。證據見team-run/image-search/BUILD_LOG.md、PROBE.md、LIVE_REPLY.md；尚待使用者實際操作確認，不寫入已驗收修正日誌。

獨立審查PASS，指出新增英文圖片搜尋trace仍為中文的小項；已依輸入語言切換簡單英文trace並加斷言，最後同六組測試仍159 passed／1 skipped（既有Tk初始化），沒有其他範圍變更。

## 2026-10-26：人物單圖入口補上Tkinter結果主視窗

上一版img_character_predict.py僅選圖對話框與終端機姓名輸出，未滿足使用者要Tkinter顯示圖片結果的要求。已改預設開啟主視窗，含匯入、門檻、重新辨識、等比例預覽、繁體中文框／名稱／分數與無結果提示；--no-gui保留文字模式。推論在worker，queue由主Tk處理，關閉等待worker，錯誤移除舊圖避免誤讀。共享renderer只新增可選空結果文字，物品預設不變，MNIST未改。

9項測試通過；另用真characters.pt及白色合成圖完成Tk預覽／無人物結果／關閉煙測，模型SHA不變。未重訓或使用實體鏡頭，未將介面測試當人物身份準確率驗收。說明同步lib/yolo/CHARACTERS.md與team-run/character-detection/IMAGE_GUI_BUILD.md。


## 2026-10-26：獨立人物圖片匯入預測入口

依使用者要求新增tranning/img_character_predict.py，沿用img_mnist_predict.py的路徑／參數、prepare_image、get_model、main與CLI區段，原MNIST程式未修改。不指定圖片時開啟選圖，亦支援圖片路徑或test_img_dir設定；只讀characters.pt，彩色BGR與EXIF／透明底處理，不套MNIST灰階，不訓練或改權重。8項新測試通過，含中文路徑、色彩、唯讀來源、模型名稱輸出、無框／取消、非法門檻與空權重；另以白色合成圖載入真實characters.pt執行，模型SHA未變。此為管線驗證，不代表人物身份準確率驗收。


## 2026-10-26：人物已訓練但best與固定門檻造成漏判

人物資料確實獨立在data/character_detection，136張中84已儲存標註；本次run d15bd93e0eed427d84a9ae9e0f74e6a7用54train／30val、5類，原characters.pt已非空且可載入。以使用者標註為真值、同類IoU>=.5配對，best在conf.35 TP5 FP0 FN26；last在conf.25 TP24 FP1 FN7，.15 TP25 FP3 FN6。這是現有val診斷，不是身份驗證保證，先前僅說輸出圖片還需訓練不適用本次實況。

已備份原characters.pt並用本run last.pt發布，SHA256與來源一致；物品資料及物品腳本權重未改。人物視窗加入best／last發布選擇（依本輪驗證預設last）、發布後清模型快取；共用標註視窗加入可設定偵測門檻，預設.25，避免寫死.35。外部更新後須重啟人物視窗，並按使用自訂模型偵測。52張未完成標註影像未刪除，也未移到物品目錄。

35項回歸通過；新增快取／門檻測試時第二次Tk建立出現invalid command name tcl_findLibrary，屬本機已見的Tk初始化現象，未改專案外安裝，另單項新程序複驗。沒有操作實體相機或再訓練。比較數據保存於output/character-detection-diagnosis/checkpoint_thresholds.json，說明同步lib/yolo/CHARACTERS.md。


## 2026-10-26：人物模型characters.pt為空檔，建立獨立資料與訓練入口

使用者要求人物特徵另存獨立.pt。實際檔案tranning/yolo_runs/custom/weights/characters.pt為0bytes，不能作為已訓練權重或訓練起點。本次保留空檔，新增lib/yolo/character_labeler.py，人物資料data/character_detection與run目錄tranning/character_detection_runs獨立；成功訓練後才備份並發布characters.pt，不改物品資料及攝影機模型路徑。35項相關測試通過，含真Tk隔離與假checkpoint檔案發布契約；沒有執行真實人物訓練或身份驗證，不能宣稱辨識品質及未知人拒絕已實作。操作見lib/yolo/CHARACTERS.md。


## 2026-10-26：最新YOLO best.pt分數過低，改用經比較的last.pt及可調門檻

使用者回報疑似門檻過高。最新602e...權重best.pt在訓練快照train+val的179個真值框上，conf=.25/.15/.10均0個預測；抽查最高分僅.0103～.0184。同run last.pt則conf=.25 TP127 FP10 FN52；.15 TP155 FP38 FN24；.10 TP164 FP67 FN15，皆以同類IoU>=.5配對。這些是訓練快照診斷，不是獨立準確率。mAP排序成績好不保證在固定門檻有足夠分數，不能只憑best檔名與類別齊全就切換；前回只確認能載入後切換不完整。

已切換同run last.pt、預設conf=.15，加入--conf與即時-/+調整，畫面顯示目前門檻；更低門檻會增加誤報，不能當品質修復。最新args使用yolo26n、batch128，85張每epoch僅1批，和先前較佳yolo11n/batch4不同，也是後續訓練需檢查因素，不能由此單獨證明所有漏判原因。未重訓、未變更標註或操作實體鏡頭。比較數據留在output/object-detection-diagnosis/threshold_88.json及threshold_88_last.json。


## 2026-10-26：88張物品資料重訓後即時腳本仍使用50張舊權重

使用者回報訓練後部分物品仍不顯示。工作區已88張、15類，最新602e16d7b6a548b9b322fec81c7e9b22訓練快照有錢包／睡眠耳機／零錢分類器，但攝影機仍指定expanded_50_transfer11（13類且無這三類）。已僅將即時腳本權重路徑改為最新best.pt，保留conf=.25及其他設定；須重啟才能載入。睡眠耳機只有2個train框、剪刀3個、存錢筒4個，鉛筆無train框，val仍只有直尺／耳機／立可帶／鑰匙各1框，因此不能宣稱換權重就能保證每類辨識。未重訓或操作實體鏡頭，亦未將人物身份辨識當成已完成功能。


## 2026-10-26：YOLO啟動時cv2匯入被中斷

使用者執行即時腳本在第一行import cv2出現KeyboardInterrupt，尚未載入YOLO或best.pt。使用同一專案.venv獨立測試：OpenCV5.0.0正常匯入，耗時0.144秒、exit0，未重現卡住。KeyboardInterrupt代表程序收到中斷訊號，不能據此判定套件損壞或模型錯誤，亦不能僅憑trace斷言是使用者按Ctrl+C。環境同時安裝opencv-python與opencv-contrib-python5.0.0.93，暫未證實與本次中斷有因果關係，未卸載或修改套件、程式、模型。建議重新執行；若再次停住再紀錄等待時間與獨立匯入結果。


## 2026-10-26：Transformer多輪串接完成，語意微調未達標

新建共用dialogue_v1、有限history預算／完整輪裁切、回答loss mask及動態padding，GUI/CLI/Web補history與隔離、/resume clear清history、/clear仍只清畫面。新checkpoint的repetition penalty只作用於生成片段；legacy維持舊格式與澄清，不切換未達標候選。資料1356train／200val／200test，val/test各152unique，原先136為估算錯誤已訂正。

初始manifest以LF字串算hash，Windowswrite_text寫成CRLF造成byteshash不符，資料本身沒被更改；已修read_bytes，保留原logical content_sha256，補實際落盤hash測試。先前評估程序在評估器更新前已import，code版本不一致的val報告只保留歷史，最終用凍結評估器全部重跑。原test實際byteshashb0f77dba42577f92866025e60bd849f1e824a66ad8a369ec505347df837d4f12不變。

兩次真實有界GPU微調：pilot_v1最佳epoch2；pilot_v2新增2048本地詞表跨度複述練習，解除凍結、lr5e-5、epoch4早停，最佳epoch1。最終test去重macro14.71%，只有姓名32.35%／物品26.47%有進展，偏好與格式0%；單輪回歸20/20跌至3/20，故不換正式gpt_chat_runs。訓練loss下降不等於理解，沒有用regex/runtime固定答案掩蓋。詳見team-run/transformer-context/EXPERIMENT.md與output/transformer-context原始報告，語意仍待改善，不寫修正成功紀錄。

初次前端測試112pass1fail為角色測試patch指到舊module，改為實際lib.components.character_browser後通過；未改角色實作。獨立工程回歸223項通過，後續新增hash／課程／REPL／legacy保護測試另通過。曾因同時代理數限制無法喚醒規劃代理，後續補強由主代理記錄決策並獨立審查，未改門檻或外部呼叫。


## Transformer本地上下文與指令遵守檢查（待修正）

使用者主要回報不記得上下文／不遵守要求。唯讀檢查確認GUI雖保存並傳遞history至smart_reply_traced，一般分支卻只呼叫transformer_chat.reply(message)，reply只編碼當句+SEP，未收到前輪；本地分支亦未自動讀取memory_store。SFT採單輪prompt/reply，尚無一致多輪角色格式。真實正式權重固定seed抽樣出現姓名答成sinco、單字限制未遵守、機器學習問題答非所問，目前output_issue未捕捉。主模型17,465,088參數／8層／6頭／384維／8000BPE／512tokens；主history最佳epoch26，訓練與驗證有落差，但程式會恢復最佳權重，不能誤稱model.pt是最後epoch。詳細[現況報告](output/transformer-audit/STATUS.md)、[獨立檢查](output/transformer-audit/REVIEW.md)與原始樣本已保存。此回合未改推論／訓練程式或權重，未重訓，不記為修正完成。


## 2026-10-25：使用者確認YOLO物品辨識改善

使用者回饋「這次的準確率很高了」，已依確認寫入[修正成功紀錄](Agent/ErrorLog/Success/2026-10-25-YOLO物品標記與小資料訓練修正.md)，包含原因、最後設定、實測比較、剩餘限制及下次排查順序。此為目前使用情境的正向確認，不代表已測得所有類別泛化準確率。整理時發現舊索引提到的Agent/ErrorFinished目錄已不存在，改依實際Agent/ErrorLog/Success目錄新增，沒有刪除歷史紀錄。


## 2026-10-23：img_nmist.py 資料載入參數名稱錯誤

- 錯誤：`image_dataset_from_directory()` 不接受 `img_size`，執行時產生 `TypeError`。
- 修正：`lib/img_nmist.py` 改用 `image_size=(img_h, img_w)`；保留既有的 `validation_split=0.2` 與訓練設定。
- oneDNN 訊息是運算後端提示，並非此次中斷原因。
- 接續檢查發現資料夾名稱拼成 `MNIST_Traning`，實際為 `MNIST_Training`，已一併修正。

> **2026-09-14 補記**：`to_do_list.md`／Claude memory 裡多處引用「ErrorLog #22」
> 「ErrorLog #23」「ErrorLog #29」，代表根目錄這份 `ErrorLog.md` 過去確實累積到
> 至少 29 號。但今天要新增紀錄時發現**這個檔案在磁碟上完全不存在**：`git log --all`
> 對它唯一的紀錄是 2026-08-17 commit `3d232f4`「刪除」（當時改成
> `Agent/ErrorLog/YYYY-MM-DD.md` 分日期檔案），但 memory 顯示 2026-09-10 那次
> 對話確實有寫入過「entry 22」——代表這份檔案後來被重建過，但那次重建從未被
> `git add`，之後不知道在什麼情況下又整份消失，也沒有留下任何 git 紀錄可查。
> **結論：1~29 號的完整內容目前已經遺失**，只能從 `to_do_list.md`（#21/#22/#23/#27/#29
> 一帶）跟 Claude memory 裡的摘要片段拼湊大概內容，原始細節找不回來了。這份檔案
> 從第 30 號重新開始記錄；建議之後每次修改都順手 `git add ErrorLog.md`，避免
> 未追蹤檔案再度整份遺失。

## 30. sinco 一般聊天（Transformer 分支）對英文輸入給出不合理的短回覆

**回報**：2026-09-14，使用者用英文跟 sinco 對話：
- 「hello, how are you?」→ 回「I'm thanking」
- 「what are you doing now?」→ 回「I'm!」

兩則回覆都文法不通、答非所問，使用者判定為「模型回應不合理」。

**現況確認**：一般聊天分支目前預設呼叫 `tranning/transformer_chat.py` 的
`reply()`（`chats.py` 的 `smart_reply_traced()` 自 2026-09-10 起改成呼叫這支，
GRU `chat_runs` 不再是預設路徑，見 to_do_list.md #27）。

**根因分析**（未實際重訓，純檢查現有 checkpoint／資料）：
1. `data/pairs.json`（609 筆）裡確實有 `"how are you" -> "I'm good, thanks for
   asking"` 這筆一模一樣的訓練資料，但沒有任何一筆涵蓋
   「what are you doing now」這類問法——英文語料本來就比中文語料稀疏。
2. `tranning/gpt_chat_runs/history.json` 最後一個 epoch（41）：
   `train_loss = 0.247`、`val_loss = 2.156`，兩者差距非常大，是典型的
   **過擬合**——8 層／384 維的 Transformer 對只有 609 筆的 finetune 資料集來說
   偏大，模型記得住見過的句子，但對「幾乎一樣但不完全一樣」（如 "how are you"
   實際上有練過，卻仍生成走樣的 "I'm thanking"）或完全沒見過的英文問法
   （"what are you doing now"）就很容易生成不連貫的短句。
3. 生成本身是取樣（`temperature=0.7`／`top_k=30`／`top_p=0.85`，非貪婪解碼），
   在分布外（out-of-distribution）輸入上取樣容易放大不連貫程度，跟
   ErrorLog #22（同一顆 checkpoint 對「我今天心情不好」有 37.5% 機率把負面
   情緒誤判成正面）是同一個「17M 參數模型的容量天花板」問題的另一種病徵，
   不是新的獨立 bug。

**尚未修正**。可能的修法（都需要 GPU 訓練時間，依
`feedback_no_unattended_long_training` 記憶規則，要先跟使用者確認要採哪個
方向、什麼時候做，不會自己擅自開一輪重訓）：
- 補更多英文日常對話的 pairs（尤其是閒聊類問法的變體），再重新 finetune；
- 降低模型容量或加強正則化（更高 weight decay／更早 early stop）減少過擬合；
- 純調整解碼參數（比照 2026-09-10 那次靠 `repetition_penalty` 不重訓就改善
  重複迴圈的做法）——但這次的病徵是「短且無意義」而非「重複迴圈」，不確定
  單靠調參能不能解決，需要先用同樣的測試句實際比較。

**相關**：ErrorLog #22（同顆 checkpoint 的負面情緒誤判，37.5% 機率）、
to_do_list.md #27（Transformer 分支整段訓練與上線過程）。

## 31. `web/backend/app.py` 的 sys.path 順序讓 `import conversation_store` 一直載到桌面版那份

**發現於**：2026-09-14，實作 to_do_list.md #36（對話紀錄依 Google 帳號分開）
時，替 `web/backend/conversation_store.py` 加上必填的 `owner` 參數後，
`pytest web/backend/tests/test_app.py` 直接炸出
`AttributeError: 'str' object has no attribute 'exists'`，traceback 指向
`lib\components\conversation_store.py:66`——不是我改的那支檔案。

**根因**：`app.py` 開頭依序
`sys.path.insert(0, BACKEND_DIR)` → `TRAINING_DIR` → `FRONTEND_DIR` →
`COMPONENTS_DIR`，每次 `insert(0, ...)` 都會把新路徑推到最前面，結果最後
`sys.path[0]` 是 `COMPONENTS_DIR`（`lib/components/`）。這個資料夾底下也有一支
同名的 `conversation_store.py`（桌面 GUI/CLI 版，函式簽名是
`create_conversation(path=None)`，沒有 `owner` 參數），Python 找模組時
`COMPONENTS_DIR` 排在 `BACKEND_DIR` 前面，`import conversation_store as
convo_store` 這幾年來實際上一直載到桌面版那份，`web/backend/
conversation_store.py` 形同沒被用到。過去兩份檔案的 API 一直保持同步
（docstring 裡也明講「改一邊要記得同步改另一邊」），所以行為看起來完全正常，
沒人發現真正被 import 進來的是哪一支。

**影響範圍**：只有這次新增的「對話紀錄依 Google 帳號隔離」功能會踩到——舊的
兩份 API 只要維持完全同步，這個 shadowing 不會造成任何可觀察的行為差異；
但這次刻意只在 `web/backend/conversation_store.py` 加 `owner` 欄位（桌面版
沒有 Google 登入概念，不需要），兩份 API 第一次真的產生分歧，才讓這個
排序問題浮出水面。

**修正**：把 `sys.path.insert(0, str(BACKEND_DIR))` 移到四行的最後一行執行，
讓 `BACKEND_DIR` 排在 `sys.path[0]`，`import conversation_store` 真的載到
這支後端自己維護的檔案。已確認 `web/backend`、`lib/components`、`tranning/`
三個資料夾之間沒有其他同名 `.py` 檔案會被這個排序調整意外影響到。
`pytest web/backend/`（41 項）、`pytest lib/components/
test_conversation_store.py`（14 項）都全過。

**相關**：to_do_list.md #36。

## 32. 長時間訓練跟著 VSCode 關閉一起中斷，中途沒有 checkpoint 可以續跑

**回報**：2026-09-14，`tranning/code_runs`（`chats.py --data code_pairs.json`）的訓練跑到一半，使用者不慎關掉 VSCode。

**確認**：
- `tasklist` / `Get-CimInstance Win32_Process` 查無任何 `chats.py` 相關 python 行程仍在執行（僅剩兩個跟本專案無關的 `AutoPush\lib\GUI.py` 行程）。
- `tranning/code_runs/progress.json` 停在 `"epoch": 2488, "epochs": 3000, "percent": 82.9, "done": false`，最後寫入時間 21:04:13，跟發現當下（21:05:24）只差約 1 分鐘，確認是被砍掉當下、不是訓練自己跑完收尾。
- 但 `encoder.pt`/`decoder.pt` 最後寫入時間是 18:40:58，比 `progress.json` 停下的時間早了 2 小時 24 分——`chats.py` 目前只有在**整個訓練跑完**時才寫一次模型權重，`progress.json`/`history.json` 才是逐 epoch 更新。等於這 2488 個 epoch（約 2.4 小時）的訓練成果完全沒有 checkpoint 可續跑，只能整個重來。

**根因（兩個疊加）**：
1. 訓練行程掛在 VSCode/Claude Code 這次工作階段底下執行，VSCode 關閉時子行程被一併終止，沒有用能獨立於編輯器存活的方式（背景服務／`nohup`＋`disown`／獨立視窗）啟動。
2. `chats.py`／`transformer_chat.py` 的權重存檔時機只在訓練全部結束時寫一次，沒有像 `progress.json` 一樣逐 epoch（或每隔 N epoch／每隔固定時間）存一次中途 checkpoint。

**原則（往後所有長時間訓練都要遵守，不只這次）**：
1. 啟動任何預期跑超過幾分鐘的訓練前，先跟使用者確認要不要現在跑、要用什麼方式背景執行——沿用既有 `feedback_no_unattended_long_training` 記憶規則，不擅自開跑。
2. 訓練腳本要能中途存 checkpoint（例如每隔 N epoch 存一次權重，取代「全部跑完才存一次」），這樣中斷後才能從最近的 checkpoint 續跑而不是整個重來。**這項程式碼修改本身還沒做**，列入 to_do_list.md #37 待辦。
3. 在確認第 2 點修好之前，任何長時間訓練都要明確告知使用者「中途中斷=整段重來」這個風險，並提醒不要在訓練跑著的時候關閉 VSCode／編輯器視窗。

**現況**：本次 `code_runs` 訓練需要從頭重跑（無中途 checkpoint 可續），使用者選擇明天再繼續，這次先不重啟。

**相關**：`feedback_no_unattended_long_training` 記憶規則、to_do_list.md #37。

## 33. 網頁前端英文對話中途被回覆成不相關的中文短句

**回報**：2026-09-16，使用者在網頁前端（web/frontend 或 web/admin）跟 sinco 對話：
- 「how are you?」→ 回「I'm good, thanks for asking」（正常）
- 接著「ok, got it.」→ 回「可以幫你看」（答非所問，且語言從英文突然跳成中文）

**確認**：`web/backend/app.py:510-549` 的 `/api/chat` 直接把使用者原文丟給
`smart_reply_traced()`，沒有任何語言偵測／過濾。一般聊天分支會走到
`tranning/chats.py:687` 呼叫 `tranning/transformer_chat.py` 的 `reply()`；
這支是純自迴歸取樣（`transformer_chat.py:624-643`，
`temperature=0.7`／`top_k=30`／`top_p=0.85`／`repetition_penalty=1.3`），
聊天分支**沒有**像 `sinco-code` 分支那樣對 `data/pairs.json` 做相似度檢索
（`code_retrieval.retrieve()` 只用在 `chats.py:658-668` 的程式碼分支），
所以「可以幫你看」不是比對錯到不相關的訓練資料，是模型對分布外輸入直接
生成出來的。

**根因**：`data/pairs.json`（609 筆）裡只有約 31 筆（≈5%）是英文提問，且
沒有任何一筆接近「ok」「got it」這種簡短應答；`pretrain()` 用的
`data/corpus_zh_starter.txt`（zh-wiki 語料）幾乎 100% 是中文。模型整體
token 統計嚴重偏中文，遇到訓練時完全沒見過的英文簡短句型，取樣時就會
偏向高機率的中文字詞組合，生成出一句語法通順但完全不相關的中文——跟
ErrorLog #30（同一顆 17M 參數 checkpoint 對分布外英文輸入生成不連貫短句）
是同一個容量天花板問題，這次多了「連語言都會跳掉」這個新病徵，之前沒記錄過。

**尚未修正**。修法方向與 #30 相同（補英文日常對話 pairs 再重訓／調整解碼
參數／降低模型容量或加強正則化），是否要現在動手、選哪個方向，需要先問
使用者，不擅自開訓練（`feedback_no_unattended_long_training` 記憶規則）。

**相關**：ErrorLog #30（同顆 checkpoint 對分布外英文輸入的病徵）、
to_do_list.md #27。

## 34. lib/opencv2/dnn.py 攝影機物件偵測只顯示最後一幀、縮排錯誤

**症狀（程式碼審視發現，尚未實機執行）**：`imshow`／`waitKey(0)` 寫在 `while` 迴圈外，只會顯示最後一幀；`net = cv2.dnn.readNet(...)` 與 `VideoCapture(0)` 縮排在 `with open(...)` 區塊內；沒有 `cap.release()`；設定檔副檔名寫成 `protext.txt`（Caffe 設定檔應為 `prototxt`）。

**修正（2026-09-21）**：`imshow` 移進迴圈並用 `waitKey(1)`，按 q／ESC 離開；縮排修正；補 `release()`；模型路徑改為相對腳本所在的 `lib/opencv2/models/`，缺檔時明確丟 `FileNotFoundError`；類別名稱改 `splitlines()`，索引越界不再 IndexError；移除每幀 `print`。

**尚未驗證**：`lib/opencv2/models/` 內沒有 `MobileNetSSD_deploy.caffemodel`／`.prototxt.txt`／`MobileNetSSD_labels.txt`，需使用者自行放入後實機測試。副檔名 `protext`→`prototxt` 是推測，若使用者檔案本來就叫 `protext` 需改回。MobileNetSSD 為現成預訓練模型，與 Rule 06 有衝突，是否保留待使用者決定。

## 35. sinco 一般聊天模型過擬合追蹤：finetune 已早停、pretrain v2 中途停在 17/25 epoch 無行程可查

**回報**：2026-09-29，`Agent/ErrorLog/2026-09-29.md` 記錄使用者要求對「一般聊天大量口吃／過擬合」做深度研究。這題本身已經是 to_do_list.md #42／#43 正在追的問題，本次只做現況診斷（`AskUserQuestion` 問過使用者，這輪範圍限定聊天模型、只診斷不訓練）。

**確認**：
1. `tranning/gpt_chat_runs_v2/history.json`（#42 的 finetune，`patience=10`）：`val_loss` 在 epoch 7 見底（3.8526），之後緩慢爬升到 epoch 17 的 3.9823，`train_loss` 同期從 2.447 一路壓到 1.112——是教科書等級的過擬合曲線；紀錄停在 epoch 17（7+10=17，`patience` 剛好打滿），檔案最後寫入時間 2026-09-22，判斷已依 early stop 自然結束，不是被中斷。
2. `tranning/gpt_pretrain_runs_v2/history.json`（#43-5，已套用「語料含 `pairs.json` 口語內容」的 BPE 修法）：`val_loss` 在 epoch 11 見底（4.4209），epoch 17 回升到 4.4808，`train_loss` 同期仍持續下降（3.52→3.01），同一種過擬合訊號在 pretrain 階段也已出現。
3. `history.json`／`model.pt` 最後寫入 2026-09-28 21:35:11，`checkpoint/` 底下最新一筆是 epoch 15（21:30:46）；`tasklist`／`Get-CimInstance Win32_Process` 查無任何跟本專案相關的 python 行程（僅兩個 `AutoPush\lib\GUI.py`，無關）。代表這個原本規劃跑 25 epoch 的 pretrain，在 epoch 17 之後、離目標還差 8 epoch 時就沒有再繼續，且發現當下（2026-09-29）已經超過 24 小時沒有動靜——研判是行程中途被中止（跟 ErrorLog #32 同一種「訓練行程跟編輯器/終端機一起被關掉」模式），不是正常跑完。好消息是這次有 `checkpoint_every=5` 安全網（#42 補的功能），只丟失 epoch 15→17 這 2 epoch 的進度，能從 `checkpoint/` 續跑，不用整個重來。

**根因**：跟 to_do_list.md #43 第 301 行已下的診斷一致——8 層／384 維／約 1746 萬參數的 Transformer，相對 `data/pairs.json` 目前 742 筆的資料量仍然偏大，不管是 finetune 還是（這次已擴充語料的）pretrain 階段，都在總 epoch 數還沒跑完前 `val_loss` 就見底反彈。目前的診斷是資料量仍是主要瓶頸，架構是否要縮小，依 #43 既定順序（資料量→斷詞→embedding→架構）要等前面步驟都試過才輪到。

**尚未修正**。這次只診斷、不訓練（使用者本輪明確選擇）。待使用者決定的選項：
- pretrain v2 要不要從 `checkpoint/`（epoch 15）續跑到 25 epoch，還是就此用 epoch 11 附近的權重當 pretrain 基底往下走 finetune；
- finetune（`gpt_chat_runs_v2`）已經因為 `patience` 自然早停，其 val_loss 最低點（epoch 7，3.8526）本身也還沒優於舊版 `gpt_chat_runs` 的最終數字（ErrorLog #30：`val_loss=2.156`），需要先用同一批測試句實測比較才能判斷是否真的優於舊版，不能只看數字；
- 是否要現在就啟動任何一段訓練，依 `feedback_no_unattended_long_training` 規則，個別詢問。

**相關**：to_do_list.md #42／#43、ErrorLog #30／#32／#33、memory `feedback_no_unattended_long_training`、`feedback_training_checkpoint_and_detach`。

**2026-09-30 修正（程式碼部分）**：`transformer_chat.py` 的 `finetune()` 新增凍結 embedding/底層 block、label smoothing、prompt 錯字擴增、pretrain 語料 rehearsal 四個抗過擬合機制（CLI 預設開前三項），細節見 to_do_list.md #44。**真實訓練尚未執行**，效果待用 `gpt_chat_runs_v4` 實測驗證後才能標記為已修正。

## 36. `git push` 一直失敗：`RPC failed; HTTP 500` / `unexpected disconnect while reading sideband packet`

**回報**：2026-09-29，使用者要求把本地領先 origin/main 的 commit push 上 GitHub，push 直接失敗。

**根因**：待推送的 3 個 commit（`模型訓練PCB`、`資料訓練擴增與報告編寫已知錯誤`、`股票預測數據資料集`）裡，前兩個一次性新增了約 **7.3GB** 的內容：
- `data/pcb/archive/PKU-Market-PCB(...)` 原始資料集（見 `project_pcb_defect_dataset_found` 記憶），數千張圖片，每張約 5-6MB；
- 6 份幾乎完全相同的 `model.pt` checkpoint（`gpt_chat_runs_v2`、`gpt_chat_runs_v3`、`gpt_pretrain_runs_v2` 各自的頂層與 `checkpoint/` 子目錄下各一份，每份約 72MB）。

`.gitignore` 本來就有排除舊版 `tranning/gpt_chat_runs/`、`tranning/gpt_pretrain_runs/` 整個目錄（可重新產生/太大），但改版後新增的 `_v2`／`_v3` 目錄名稱沒有同步補上規則，才會被意外整包 commit 進去。單次 push 7GB+ 遠超過 GitHub HTTPS 服務穩定處理的範圍，才會回 HTTP 500。

**修正過程**（依使用者要求：最多 5 次修正）：
1. `git config http.postBuffer 1048576000` + `http.version HTTP/1.1` 後重試 → 失敗，仍是同樣錯誤。
2. 單純重試（排除暫時性問題）→ 失敗，確認不是網路瞬斷。
3. 用 `git diff-tree`／`git cat-file --batch-check` 精確定位是哪個 commit、哪些檔案造成 7.3GB，向使用者說明兩條可行路線（從 commit 中移除大檔案 / 改用 Git LFS / 先不處理只記錄），使用者選擇「從尚未推送的 commit 中移除大檔案」。
4. 先建立備份分支 `backup-before-cleanup-20260929`（指向重寫前的原始 3 個 commit），再用 `git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch ...' -- origin/main..HEAD` 只重寫這 3 個本地尚未推送的 commit，把 `data/pcb/archive` 與 6 份 `model.pt` 從歷史中移除（origin 上沒有這些 commit，重寫不影響任何共享歷史）。
   - **副作用**：`filter-branch` 重寫後會用新 HEAD 覆蓋工作目錄，導致這些大檔案連硬碟上的實體檔案都被砍掉，不只是取消追蹤——不是預期行為，發現後立刻用 `git checkout backup-before-cleanup-20260929 -- <paths>` 把檔案救回硬碟（再 `git reset` 取消暫存，維持未追蹤狀態），沒有實際遺失資料，但這是這次操作裡最大的風險點，下次做同類 `filter-branch` 清理前應該先預期這個副作用。
   - 在 `.gitignore` 補上 6 個 `model.pt` 精確路徑與 `data/pcb/archive/` 整個目錄，避免以後又不小心重新加回版控。
5. 重新 `git push origin main` → 待推送內容降到約 18MB，push 成功（`e5fa380a..73071297`）。

**已修正**。備份分支 `backup-before-cleanup-20260929` 目前仍保留在本地（未刪除），供之後需要回頭核對原始歷史時使用；它只存在本機，不會被 push 上 GitHub。

**相關**：`project_pcb_defect_dataset_found`（PCB 資料集來源）、`project_transformer_chat_retrain_v2` / `project_nlp_strengthen_plan`（`gpt_chat_runs_v2`／`gpt_pretrain_runs_v2` 這兩顆 checkpoint 的訓練脈絡）。

## 37. /character 選了角色，回覆卻還是 sinco 本人（角色性格完全沒生效）

**回報**：2026-09-30，使用者回報「人物的學習模擬個性不夠接近一般人類的回答，要能夠依使用者描述的性格進行對話」。

**根因**（查程式碼確認，不是推測）：2026-09-10 把 `chats.smart_reply_traced()` 的一般聊天分支從 GRU 換成 `transformer_chat.reply(message)` 時，這一行**沒有帶 `out_dir`**——GUI（`conversation.py`）／CLI（`cli.py`）／網頁（`web/backend/app.py`）切換角色時改的是 `out_dir=character_chat_runs`，但一般聊天分支完全不看它。結果從 9/10 起：選了「周柯宇」，聊天視窗標籤顯示周柯宇，實際回覆卻是 sinco 一般模型；角色卡的描述、氣質分數、你寫的對話範例一個都沒被用到。另外兩個既有限制讓問題更明顯：(a) 舊的 `character_chat_runs` 是 GRU 死記 21 句（loss 0.0006），沒背過的問法本來就答不好；(b) 所有角色共用同一顆 checkpoint，新角色只寫描述完全沒有作用。

**修正**：新增 `tranning/persona_chat.py`，`smart_reply_traced()` 新增 `character` 參數，GUI／CLI／網頁三個前端都把目前角色名字傳進去。角色回覆流程：你寫的角色範例（相似度比對，會先拿掉角色名字/暱稱）→ 沒有範例時讓 sinco 產生 6 個候選、依性格打分挑一個再改寫成角色語氣（拿掉 AI/sinco 身分句、只留口語一兩句、依性格補語尾/稱呼/表情，高冷型則反過來精簡）→ 模型沒訓練時用依性格寫好的備用短句。性格來源 = 角色卡滑桿分數 + 從自由描述文字抓到的性格關鍵字，所以只寫描述的新角色也能用。詳見 to_do_list.md #45。

**已修正（程式碼）**，`pytest tranning/test_persona_chat.py`（13 項）、`test_chats.py`、`lib/components/test_cli.py`、`web/backend/tests/` 全過。尚待你在 GUI 實際用 `/character` 切換角色聊幾句確認體感。



## 2026-10-18：灰階分類器特徵層未從自己的資料學習

檢查發現 `image_classifier_bnn.py` 原本使用外部 ImageNet 預訓練 MobileNetV2，且預設凍結 features；單純加灰階前處理不能讓卷積特徵層學習。新訓練改用自建 FeatureCNN，全卷積層可訓練，config 記錄架構；舊 checkpoint 保留讀取分支但禁止下載權重。新增橫線／直線合成測試，驗證三階段卷積梯度非零、權重更新及 loss 下降；含既有分類流程共 7 項測試通過。尚未執行真實資料重訓，待使用者驗證實際效果，未登記為功能驗收完成。


## 2026-10-18：特徵輸出範例圖片路徑不存在

使用者直接執行 `--features "圖片路徑.jpg"`，因該文字是佔位範例而非實際檔案，Pillow 拋出 FileNotFoundError。文件改提供專案現有驗證圖片範例；export_features 先檢查圖片是否存在，再載入模型，CLI 將路徑與圖片格式錯誤顯示為參數提示，不再輸出整段 traceback。圖片路徑相對於執行時目錄，並非 checkpoint 目錄。保留模型及原圖。


## 2026-10-18：一般問答偏題、搜尋污染與前端狀態

使用者案例為「你懂得股票投資嗎」「什麼是交叉商損失」「名詞解釋什麼是RAG」。直接既有Transformer推論已重現偏成程式邏輯／英文文法（team-run/BASELINE.json）。一般分支未使用多輪歷史，且過寬路由讓故事進程式、能力問答進行情。加入保守雙語規則與有限輸出檢查，仍不代表模型語意能力提升。

獨立審查抓到「台股／天氣的意思是什麼」仍誤進即時工具，已改定義意圖優先並加反例；英文禮貌搜尋不再被能力gate略過。公開DDG回202驗證頁，Bing RSS能取得索引但RAG實際回政治人物內容，因此加入明顯詞面不符過濾。此次測試新增的錯誤候選（id 6ebce3ff）已移除，保留既有候選及pairs。RAG／交叉熵精確本機詞條附原始論文／官方文件，明示日期、非即時／非生成，不宣稱搜尋或泛化成功。

前端原本裸來源URL不可點、hidden指令選單被display:flex覆蓋而留下14px空白條，已修格式與CSS；新增IME、防重送及錯誤復原。第一次連結測試發現trim分號會破壞HTML entity，已保留分號。整合測試最初引用錯誤測試路徑而未跑，改正後執行；測試stub有未await coroutine警告，非瀏覽器實測失敗，另有既有套件deprecation警告。

重要未完成項：真實24筆離線驗收顯示自然中英文未見題仍不通順／偏題，記錄於team-run/chat_eval_offline.jsonl。未進行長時間訓練或替換權重，正常人類／母語品質不視為完成，見to_do_list #46。此處記錄問題與工程處置，尚待使用者實際驗收。

補充：既有HTTP測試曾寫入真實usage.json的testclient測試帳；已增加autouse暫存用量fixture並還原該測試帳原始紀錄，保留其他使用者用量。20項HTTP測試重驗通過。

## 2026-10-18：聊天版型重整的瀏覽器驗證問題

重整實際 admin PyScript 前端時，手機規則 `.btn-export-md span` 同時隱藏文字與包住 SVG 的 span，使匯出按鈕外觀變成空白。改以 `.export-label` 限定文字標籤，並補充按鈕 ARIA 名稱及最小點擊區，待使用者驗收整體介面。

瀏覽器驗證腳本首次等待「後端已連線」可見而逾時；該狀態已收在預設關閉的系統狀態 details 中，文字存在但不可見。檢查條件改等待節點存在。此為測試條件與新布局不符，非 API 連線失敗。瀏覽器測試使用實際 PyScript DOM 與模擬 API，未代表真實模型、相機或通知服務驗收。

## 2026-10-18：時區與 Unicode 練習的 Windows 測試編碼

CLI 子程序 smoke test 首次以 UTF-8 解碼系統預設編碼的輸出，發生 UnicodeDecodeError。測試呼叫改用 Python `-X utf8` 後，emoji、結合字及 200 碼位限制檢查通過。從專案根合跑 `web/backend/tests/test_frontend_chat.py` 與 `web/admin/frontend/src/components/test_render_markdown.py`，共 8 項通過；後者 mock 排程出現 4 個 coroutine 未 await 警告。單跑前者則是 4 項通過、無警告；兩者範圍不同，不代表使用者驗收完成。

## 2026-10-19：瀏覽器時區別名的測試斷言

四時區驗證時，Chromium 的 Intl API 將輸入 Asia/Kathmandu 回報為別名 Asia/Katmandu；原測試要求字串完全一致而失敗。測試改接受該別名，UTC+05:45 檢查保留，四時區重跑通過。產品維持顯示瀏覽器真實回傳值，未將別名當成位置證據。

## 2026-10-19：本機 IP 國別資料下載需要來源標頭

初次透過 urllib 下載 DB-IP 公開 Country Lite MMDB 壓縮包時 HTTP 403；供應商公開下載頁可正常讀取，確認相同官方連結後，為下載請求加入明示 AI-module User-Agent 與下載頁 Referer 即成功。已解壓並用 maxminddb 讀取資料庫，114.114.114.114→CN、1.9.1.1→MY、168.95.1.1→TW；來源與 SHA256 記於 team-run/region-signals/database-download.json。此為資料庫讀取 smoke，不代表已驗證真實使用者所在地或風控準確率。

補充（本輪開發過程）：初次套用內部評估 hook 的 patch 曾命中通知處理區段，已移除該誤置區塊並改置 /api/chat；目前搜尋 assessment 僅在聊天流程，50 項規則／HTTP／前端回歸通過。此記錄為開發問題與驗證，不代表使用者已驗收。
測試開發補充：初版測試曾有括號語法錯誤，另因 app.py 採裸 import region_signals，mock 若套用 package import 副本不會影響實際API；已修語法並改對 app 模組所持參照做 mock，最終50項通過。

## 2026-10-19：OpenCV 讀圖與 Request.client 的可空型別

`lib/img_convert_gray.py` 的 cv2.imread 可能回傳 None，直接傳進 cvtColor 造成型別警告及執行錯誤；讀圖後加入 None 檢查，失敗時顯示路徑／格式提示。保留原有註解與其他範例內容。

`web/backend/app.py` 的 _resolve_client_id 原先直接讀取 request.client.host，client=None 會 AttributeError。首次驗證仍重現該行，重新核對並套用後，統一轉呼叫既有 _client_ip，缺來源時回 unknown。新增缺來源／正常來源兩項回歸，後端25項通過；讀圖失敗防護與實際BGR轉RGB smoke通過，未開啟攝影機。本環境未安裝Pyright，未宣稱完成靜態檢查器重跑；尚待使用者IDE確認。保留兩項既有第三方deprecation警告。

## 2026-10-19：安裝檔超過 GitHub 單檔上限，導致 push 失敗

使用者回報 `app/install/install.msi`（161,636,352 bytes，約154.15 MiB）超過普通Git的100 MiB限制，分批push不能繞過單檔限制。將origin/main以外的6筆未推送提交轉為Git LFS，加入.gitattributes；本機原始歷史另保留於backup-before-lfs-20261019，不推送此備份分支，不強制覆寫遠端歷史。遷移後執行git lfs checkout還原本機實體安裝檔，SHA256與遷移前相同，並檢查待推送普通blob已無超限檔案。

處理時發現LFS遷移也更新了指向同一tip的本機備份ref，已將備份明確恢復至原提交1abc5f64842812ea0185fbcdf89a8899b7bde0aa。一次Git物件檢查遇到PowerShell環境CP950解碼錯誤，改用bytes擷取與UTF-8解碼後檢查成功。大檔先透過LFS上傳，再依可fast-forward的提交分批推送；遠端是否同步以推送後核對為準。


## 2026-10-19：MNIST 舊版匯入路徑與未完成語法

`lib/pytensorflow/main.py` 的 `from tensorflow.examples.tutorials.mnist` 缺少同一陳述式內的 `import ...`，下一行 `import input_data` 會被視為另一個頂層模組匯入；檔尾另有未完成的 `def de`。舊教學 MNIST input_data 不能直接套用現行 tf.keras 資料集介面，應改用 `tf.keras.datasets.mnist.load_data()`。實查專案 .venv 未安裝 tensorflow／keras；尚未安裝套件、下載 MNIST 或修改使用者未完成的範例程式，只記錄診斷。


## 2026-10-19：MNIST 函式參數作用域及新舊 TensorFlow 介面混用

`lib/pytensorflow/main.py` 的 conv2d() 只有空 return，後續 tf.layers.conv2d 未縮排而落在模組層，input_layer／filters 超出函式參數作用域；另殘留未匯入的 input_data、one_host 拼字及 pool_size=[2,25]。沿用原四個函式改為 tf.keras.layers，MNIST 僅透過 keras 載入一次，保留舊教學55,000訓練／5,000驗證、4,000測試及flatten／one-hot格式。Dropout需在訓練時傳training=True，CNN使用前需將flatten影像reshape為(N,28,28,1)。語法與return作用域檢查通過；.venv仍未安裝TensorFlow，未宣稱通過真實TensorFlow推論或資料下載。初次patch因檔案空行上下文不符而未套用，重新讀取後以較小範圍修正，未修改其他筆記。

## 2026-10-19：MNIST 輸入與展平變數名稱不一致

lib/pytensorflow/main.py 定義 x_input 卻使用 input_x，定義 flat 卻使用 flat_0，造成未定義名稱。統一為 x_input 與 flat_0，並以 tf.keras.Input、Reshape、Flatten 取代混入的 TF1 placeholder 與 reshape，沿用前處理的 0~1 資料，避免重複除以 255。Keras 標籤由後續訓練傳入，不建立 y_output placeholder。語法及名稱檢查通過；專案 .venv 未安裝 TensorFlow，未執行真實模型、資料下載或訓練。

## 2026-10-19：MNIST 損失函數使用未定義 y_output 與 TF1 介面

lib/pytensorflow/main.py 已改為 Keras Input，新增的 TF1 損失／優化器／指標仍引用已移除的 y_output，另有 softmax_cross_entopy、lavels 拼字錯誤。改用 Keras Model.compile、CategoricalCrossentropy(from_logits=True)、Adam(learning_rate=0.000095)、CategoricalAccuracy；真實 one-hot 標籤由後續 fit 的 y_train 傳入。輸出 logits 移除 ReLU；Dropout 改 training=None，讓 Keras 依訓練／推論階段控制。語法及設定靜態檢查通過，.venv 未安裝 TensorFlow，未執行訓練或宣稱實際準確率。


## 2026-10-20：照片方向與灰階修圖示範的能力邊界及執行紀錄

既有影像分類／CVAE 不代表已具備任意照片的方向推理或文字語意修改。本次新增獨立示範 `tranning/photo_edit.py`、`lib/photo_edit_demo.py`，有限精確指令可立即執行；方向 CNN 與灰階文字殘差 CNN 從零訓練。沒有方向權重時不自動旋轉，低信心時保持方向；灰階外另存原色度，不宣稱能由灰階恢復原始色彩。真實方向泛化、視角理解、任意文字物件修改仍未驗收，列入 to_do_list.md #47。

執行中發現／處理：
- 根 README 的首次 apply_patch 因上下文不吻合而未套用（沒有修改檔案），改用 UTF-8 追加獨立章節，保留既有內容。
- 檢查時發現手動指定角度仍先讀方向 checkpoint，會受無效權重路徑阻擋；改為手動角度優先分支。
- 純添加標記原先仍做 YCbCr 往返，可能改動未修改位置的 RGB；調整為沒有亮度操作時直接複製原圖。
- 工具擷取 Python 終端輸出時曾出現中文字碼顯示不符；UTF-8 原始檔與 JSON 正常，驗證程序設定 PYTHONIOENCODING=utf-8，未讀寫 .env。

整合驗證：`.venv/Scripts/python.exe -m pytest tranning/test_photo_edit.py -q --basetemp=team-run/photo-edit/integration-tmp`：4 passed。另有企鵝實拍逐步輸出與一次 epoch 合成 smoke；測試／合成權重僅證明管線可跑，不能作真實準確率或功能驗收完成的依據。獨立審查結果另見 `team-run/photo-edit/REVIEW.md`。


照片示範獨立審查補充：發現預設指令添加圓形固定於 (40,40)，使寬或高不超過 40 的合法照片失敗。核心與 CLI 改共用不含座標的亮度／銳化預設，保留顯式圓形操作；新增 1×1、12×30 的 CLI 回歸驗證與越界座標拒絕測試。最終整合測試 6 passed（3.95s）。建置補充文件時另有一次工具 JavaScript 字串包裝 SyntaxError，shell 未執行、檔案未改，重試完成。

另產出 `output/photo_rotation_example/` 實拍人工旋轉案例，輸入逆時針 90°，指定校正 270°；獨立逐像素比較確認校正圖等於原始照片。此為人工指定角度的流程示範，並非模型自動判斷成果。一次 epoch 合成方向 smoke 驗證準確率為 0.25，權重只用於管線測試，不能投入正式照片方向判斷。


## 2026-10-20：電路圖邊緣處理示範

使用者新增 `lib/img_edge.py` 練習檔以 `cv2.imread()` 讀 BGR，卻使用 COLOR_RGB2GRAY，且 Canny 迴圈未存圖；本次沿用其架構修正為 COLOR_BGR2GRAY、high 拼字與實際輸出，加入 photo／circuit CLI。核心新增共用 extract_edges：電路 Canny 使用3×3輕微去噪，另外對原灰階作 Otsu 反向二值化保留線寬，不用閉運算／膨脹以免隨意接線。電路預設不做亮度／銳化。既有 KiCad 範例渲染有近似符號與重疊標籤，不作清晰教學例，另外用 schemdraw 產生 RC 圖；未修改原 KiCad、SVG 或渲染器。

驗證 `.venv/Scripts/python.exe -m pytest tranning/test_photo_edit.py -q --basetemp=team-run/photo-edit/circuit-test-tmp`：8 passed。新增細線、單像素間隙、預設電路不改色、門檻檢查、練習 CLI 三圖輸出與拒絕覆寫。實際跑 RC 圖兩種 CLI 並檢視 preview。此為影像前處理，尚無元件或導線連接理解，黑底圖與強陰影掃描尚待前處理，不登記為完整功能驗收。首次檔案搜尋有 rg 無匹配回傳1，非 Python 或測試失敗。


## 2026-10-20：麵包板實拍不能沿用白底墨線假設

使用者要求支援杜邦線、電阻等麵包板實拍，既有 circuit Otsu 僅適用白底深線，不能當作實拍元件遮罩；灰階也無法保留線色及色環供後續辨識。新增 breadboard 模式保留彩圖，雙邊保邊去噪與 Canny，額外保存非語意 HSV 彩色區域及邊緣疊圖，不自動補線、不輸出元件／導通判定。尚待使用者確認後續層級與自有照片。

尋找官方實拍時 SparkFun 教學頁由 web 工具取得403；Adafruit教學頁的 urllib存取也回HTTP403，未持續繞過限制，改採搜尋結果所列 Adafruit公開CDN產品圖，成功保存為 output/breadboard_example/source.jpg 並記錄來源與SHA256。照片已目視確認為含麵包板、杜邦接頭線、電阻、LED及開發板的實拍，不是原理圖。此照片僅本機處理示範，未用於模型訓練。

驗證 `.venv/Scripts/python.exe -m pytest tranning/test_photo_edit.py -q --basetemp=team-run/photo-edit/breadboard-test-tmp`：10 passed。兩入口實拍執行完成；目視檢查 preview，原色保留、邊緣與彩色區域可見。新增斷言：預設原彩圖像素與來源相同、黑線未出現在飽和遮罩但保留於原圖、RGB色序正確、沒有誤產生原理圖墨線遮罩；報告明確註明元件／連接not_implemented。仍未驗證多種類／多角度真實電路辨識率。

麵包板共用前處理獨立審查通過（BREADBOARD_REVIEW.md）；再次獨立驗證十項測試、小尺寸、無效參數與不覆寫，實拍兩入口原色逐像素相同且來源SHA256不變。此審查不代表完整元件或接線能力通過驗收。


## 2026-10-20：地區封禁需求盤點

讀取前端註解所指 web/backend/auth.py 時確認該檔不存在；實際 app.py 以來源 IP 識別，不能當成穩定帳戶身分。另一靜態前端已有公開 Google Client ID，但實際後端提供 admin/frontend，未接帳戶驗證。region_signals.py 為 observe_only，尚無已定義文字封禁門檻或低誤判率資料。本次先記錄規格、隱私草案與模型差距時間表，待使用者確認身份來源、介面與文字門檻後才實作封禁；未觸碰 .env，未宣稱修復或上線完成。


## 2026-10-20：自有接線資料匯入與標註缺口

正確來源為Downloads/電路資料集，先前使用者誤傳的專題測試結果排除。新工具lib/breadboard_dataset.py／tranning/breadboard_dataset.py唯讀整理照片影片，實跑加--hash-videos產生15來源（8照片7影片）、29資產（8原照21抽幀）、errors0與全來源SHA256，原檔不搬移不覆寫。大影片僅有界抽樣，時間是index/FPS估計，不是PTS；尚無元件、腳序或接線真值。端點／連線／孔列模板的空白是未知，不能當斷路或負例。watch技能的預設setup會讀寫專案外設定及.env、可啟動外部模型，與使用者規則衝突，因此僅依本機抽幀方法以cv2/PIL處理，不執行設定與轉錄。

新增工具測試3 passed（合成MJPG實際寫入解碼，無skip），舊影像測試待整合複驗；真實3近拍已以原解析度處理。未發生新程式／解碼錯誤，沒有啟動正式訓練或聲稱接線辨識完成。使用者已一次收到預期電路與三腳元件資料補問，等待期間完成資料前置，不捏造標註。


資料整理獨立審查補充：原本來源stat/SHA256在單檔try外，模擬PermissionError會整批中止且無manifest；縮圖重讀來源失效亦可能丟失紀錄。已將stat/hash納入逐檔錯誤處理並繼續其他來源，manifest在縮圖前先落地，縮圖逐張失敗改佔位並記error。公開函式原本接受3.5／True作抽樣數，改為真正int且排除bool、範圍1..32，非法值不建輸出。加入來源拒讀／另一張成功、縮圖失效與型別回歸後，新增工具7項＋既有影像10項共17 passed（3.95s）。先前審查一度使用不存在的lib/test_img_analyize.py導致pytest無測試可跑，隨即改正測試路徑；屬操作錯誤，非程式失敗。既有真實15來源／29資產／0錯誤輸出與空模板未更改。


## 2026-10-24：MNIST 自訂測試圖片路徑

lib/img_nmist.py 的 test_img_dir 原為 Path(__file__)，指向程式本身，且下方單張預測另行寫死 Test_data/5.jpg，未使用該變數。已將預測路徑接至 test_img_dir，並依邊框底色將小畫家白底黑字反轉為 MNIST 黑底白字；模型內已有 Rescaling(1/255)，不再對輸入重複正規化。新增空白／極低對比圖片拒絕，輸出類別與 softmax 機率。未執行 30 epochs 正式訓練或評估使用者手寫圖片準確度。


## 2026-10-24：獨立 MNIST 數字輸出入口

依使用者澄清新增 tranning/img_mnist_predict.py，不再修改 lib/img_nmist.py（保留使用者後續更新的 mnist_test/p1.png 路徑）。新入口使用同一圖片路徑，無圖表、無機率陣列，只向 stdout 印一個數字；找不到快取模型時從本機 MNIST 訓練 5 輪並保存於 tranning/mnist_runs/digit.keras。已以合成資料跑過一輪訓練、模型保存／載入、黑白底前處理、空白拒絕與單數字輸出，並以 SHA256 確認原檔未變；未執行真實 60000 張完整訓練，未宣稱使用者圖片辨識準確。檢查時 Keras 提示 Dataset 自行洗牌，因此 fit 明確設 shuffle=False。TensorFlow 仍可能將底層環境訊息寫入 stderr，與數字 stdout 分開。


## 2026-10-24：小畫家細筆畫縮圖後誤判為 1

檢查真實 p1/p2/p3 圖片與現存 digit.keras：原前處理縮成 28x28 後，圖片最高亮度僅 51～61，筆畫近乎消失；模型抽查 MNIST 0/1/3/7 可辨識，三張使用者圖片卻皆誤判 1。此外 model_path 存在即載入，修改 epochs 不會重訓。僅修改獨立 tranning/img_mnist_predict.py：稀疏筆畫先依原尺寸加粗、補黑邊後縮圖，並新增 --retrain 明確重新訓練。保留使用者 p3.png 與 epochs=30 設定，未修改原課本檔、圖片或已訓練模型。三張真實圖片已實測 p1=7、p2=7、p3=2；main 標準輸出精確為 2 換行，模型 SHA256 不變。此結果不代表所有手寫字體準確度。

## 2026-10-25：物品標註工具測試清理與中文字型下載

新增標註工具首次測試出現 `_tkinter.TclError: can't invoke "winfo" command: application has been destroyed`：測試在 `app.close()` 銷毀 Tk 後又查詢視窗。已將測試清理改為安全處理重複 destroy；屬測試清理錯誤，未刪除或修改使用者影像。

檢查已安裝 Ultralytics 8.4.90 原始碼發現 `check_det_dataset()` 即使 `plots=False`，仍因中文類別呼叫中文字型下載檢查。新物品微調入口只在 `model.train()` 範圍暫停該字型檢查、保留資料驗證，結束或例外均還原；畫面中文由 Tk 顯示。未修改套件檔，升級套件後需重新確認相容性。驗證與限制見 [建置紀錄](team-run/object-labeling/BUILD_LOG.md)；尚未正式訓練或驗證實體攝影機，不作使用者驗收紀錄。

獨立審查首次整組測試為 25 passed／1 failed，第三個 Tk 建構瞬時找不到 `ttk/notebook.tcl`，發生在 `tk.Tk()`；單項重跑通過，再完整重跑為 26 passed（2.61s）。未修改專案外 Python 安裝；保留此環境現象供重現追查。審查判定與獨立多類座標／負例檢查見 [審查紀錄](team-run/object-labeling/REVIEW.md)。

## 2026-10-25：YOLO 即時畫面唯讀導致 FPS 繪字失敗

使用者載入自訂 best.pt 後，`lib/yolo/Instant Image Reconginizion.py` 在 `cv2.putText()` 出現 `img marked as output argument, but provided NumPy array marked as readonly`。直接原因為 `results[0].plot()` 回傳唯讀影像，OpenCV 5.0.0 拒絕原地寫入；不是權重路徑錯誤。改為 `plot().copy()` 取得可寫副本，並將 FPS 繪製移至 imshow 之前，保留使用者 best.pt 路徑與原有註解。

以真實 cv2.putText 重現唯讀陣列錯誤，再用假 YOLO／攝影機執行整支腳本，確認複製後可繪字、顯示前已有 FPS、來源影像未修改、按 q 後釋放鏡頭。未占用實體攝影機或驗證辨識精度；待使用者實機確認，不寫入已驗收修正日誌。

## 2026-10-25：自訂物品權重沒有偵測框與繁體中文繪製

檢查 ff12324270144ba8833cc34af3a25f3f/best.pt，模型類別確有橡皮擦等13個繁體中文名稱。其訓練快照只有11張訓練圖，每個已出現類別各1個框；類別1與11沒有訓練框。驗證只有1張圖、2個框（直尺與耳機）。以真實權重在GPU抽查2張訓練圖與1張驗證圖，conf=0.001下最高分別約0.001226、0.004405、0.001295，全部低於即時程式0.35門檻；這解釋抽查時沒有框，不能以降低門檻或畫人工真值假裝辨識正確。

新增 lib/yolo/detection_overlay.py，使用本機Windows微軟正黑體以Pillow繪製模型預測框與繁體中文名稱；即時程式改呼叫此函式，沒有預測框時明確顯示中文提示。合成預測驗證中文繪製、框線、唯讀來源不變及可寫輸出，輸出示意在 output/object-detection-diagnosis/chinese-label-check.png；不是模型真實辨識成果。未重訓或操作實體鏡頭。下一步需要補各類多樣照片及獨立驗證資料，既有模型品質尚未通過驗收。

## 2026-10-25：新增標註與重訓後攝影機仍載入舊模型

使用者已提供30張工作區影像，29張已儲存：train 27張／57框，val 2張／4框，另1張未確認；新模型 f68b9a8872ae4be897db2c5fc417cf4c 使用這份快照、batch4、epochs50。但即時腳本仍指向舊的 ff12324270144ba8833cc34af3a25f3f（11張train），已僅更新權重路徑為新模型。新權重GPU抽查2張train、2張val，最高分別0.2109、0.0636、0.0233、0.1645，均未達0.35；不能宣稱換路徑就能準確辨識。未改門檻、未重訓或操作實體鏡頭；先前只依舊資料提出增加照片的建議不完整，應先核對實際模型與資料快照。

## 2026-10-25：物品標註工作區刪除功能

依使用者要求移除1張尚未確認、沒有框且非背景負例的照片（e6a144232ad8f861e8e768fabcaafdb0561130d0e9e2c3cf31978c104aa5beca）；工作區30→29張，原圖副本及原annotations.json保留在data/object_detection/trash內，已確認備份存在。新增「刪除未標註照片」「刪除類別」「更名類別」，類別刪除同步調整ID並讓受影響照片待重新確認，更名保留框；既有匯出與模型不變。30項相關測試通過，未操作實體鏡頭或重訓。操作文件首次套用patch因預期文字不是完整行而失敗，無修改發生，改以標題定位後成功。詳見team-run/object-labeling/DELETE_CHANGES.md；未宣告使用者驗收完成。

## 2026-10-25：小資料訓練更新不足與50張加強比較

舊模型即使conf反覆降低仍誤判／漏判。確認27train每epoch7批，batch4與預設nbs64配合會約16批累積更新、warmup至少100步；先前50epochs有效更新少，少樣本多類與強增強也是因素，不能只歸因單一參數。使用者期間擴充為50張且全部儲存，另建48train102框／2val4框快照；舊模型在此快照conf.25 IoU.5僅TP1 FP1 FN101。

完成有界YOLO26更新設定比較及最後本地YOLO11凍結特徵150epoch實驗；最終best在同快照train TP101 FP21 FN1、val TP4 FP1 FN0。訓練圖指標不等於泛化精度，驗證只有4框且無橡皮擦。使用者截圖回放best能找橡皮擦82.01%／直尺42.12%，單顆橡皮擦70.76%，仍漏指甲剪。選expanded_50_transfer11/weights/best.pt，保留舊模型，GUI/CLI同步較佳配方、加入缺類警告；即時conf改為評估用.25並印實際模型路徑。詳見team-run/object-labeling/TRAINING_COMPARISON.md及output/object-detection-diagnosis原始報告，尚未做實體鏡頭驗收，不記為完整物品辨識已修好。

測試中有瞬時Tk缺listbox.tcl／notebook.tcl，單項及完整33項重跑通過，不修改專案外安裝。根代理真權重GPU搭配假鏡頭執行整支即時腳本，中文繪製、可寫陣列與q釋放通過。臨時接觸表腳本首次因PIL.ImageOps未匯入失敗，改為按圖片數預先配置畫布後成功；完成程序後一次輪詢回報Unknown process id，已由結果檔確認正常結束。更新攝影機patch因使用者同期將conf從.15改.05而比對失敗，重讀後僅更新模型與驗證門檻，其餘修改保留。

## 2026-10-24：GitHub 專案快取清理

使用者明確授權清理 C:\Users\roy\Documents\GitHub 內所有專案。盤點排除 Git、虛擬環境、依賴、建置輸出、模型、資料集與 .env，僅處理 __pycache__、.pytest_cache、.mypy_cache、.ruff_cache 與個別 pyc/pyo/系統縮圖快取；每個目標確認位於授權目錄內、無路徑連結、未被 Git 追蹤。永久刪除命令遭自動安全審核以 blocked by policy 拒絕，未執行；改採可復原的 Windows 資源回收筒。

共 11 個專案、70 個快取項目、553 個檔案、7459395 bytes（7.11 MiB）已移至資源回收筒，0 失敗，逐項確認原位置不再存在。詳細盤點、驗證名單與結果在 output/cleanup-audit/。沒有清空資源回收筒，因此不宣稱已釋放磁碟空間；既有模型與成果檔保留。
# 2026-10-26 字詞語意：預訓練資源下載受限

- 獨立審查找到詞 span 將字面 `[MASK]` 等控制 token 當一般詞的問題（M1）；已同步在保留與完整 token offsets 排除控制 special IDs，保留一般未知字 UNK，並新增純控制／混合範圍測試。詳見 `team-run/word-sense/REVIEW.md`，本輪未因此修改模型權重。
- 回歸測試使用專案內已建立的父目錄後完成：原 Transformer／embedding 20 passed；第一次指定的 basetemp 父路徑未存在曾引發 WinError3，已建立父目錄解決。

- 測試環境另遇預設 `AppData/Local/Temp/pytest-of-roy` 與舊 `.pytest_cache` 的 `WinError 5` 權限問題，15 項既有回歸測試在 fixture 建立時受阻，非模型斷言失敗。改用專案內專用 `--basetemp output/word-sense-tests/...` 並 `-p no:cacheprovider`，不修改專案外權限或檔案；新測試也採此方式。

- 嘗試以專案 `.venv` 的 `huggingface_hub.snapshot_download` 將 `BAAI/bge-small-zh-v1.5` 存入專案 `models/`，HF_HOME 亦限本專案。
- 連線在 repo_info 階段遭執行環境拒絕：`httpx.ConnectError: [WinError 10013]`，後續為 `LocalEntryNotFoundError`，未完成權重下載。
- 保留離線載入入口與可由使用者執行的明確下載指令；不繞過網路限制，不將隨機測試模型宣稱為預訓練模型，也不宣稱真實消歧品質已驗證。


## 2026-10-26：字詞語意資料建置工具問題
資料Builder首次用PowerShell管線傳送含中文Python程式，系統編碼將文字轉成問號；已改為UTF-8原始檔build_fixtures.py重建六個word_sense JSON，新增span、無問號與SFT對齊測試。首次rg對tranning/gpt*路徑在Windows回報os error123，改明確檔名。三個新測試通過；pytest cache因WinError183無法建立，僅快取警告，不更改專案外環境。未執行模型訓練。
