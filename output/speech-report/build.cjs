const fs=require('fs');
const {Document,Packer,Paragraph,TextRun,Header,Footer,PageNumber,AlignmentType,HeadingLevel}=require('C:/Users/roy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/docx');
const font={ascii:'Times New Roman',hAnsi:'Times New Roman',eastAsia:'標楷體',cs:'Times New Roman'};
const blocks=[
['title','自建模型語音辨識：台灣華語、閩南語與中國大陸口音'],
['p','現況盤點、資料規劃與協作審查報告｜2026年11月3日｜版本 1.0'],
['h','一、執行摘要與範圍'],
['p','本報告針對 AI-module 的自建語音辨識能力，區分已存在的程式與權重、已保存的訓練紀錄，以及尚未驗證的新語言能力。第一階段以台灣華語短句聽寫為基準，新增台灣台語（閩南語的一部分）與中國大陸普通話／地方口音的資料及評估規劃。本輪沒有執行新訓練，也沒有修改聲學模型；報告中的改善方向是待實驗的工程方案，不能視為已經提升準確度。'],
['p','「聽懂」至少包含兩層：語音轉成忠實逐字稿，以及理解逐字稿中的意圖、數值與條件。前者辨識成功不保證後者推理正確；後者偶爾猜對，也不能掩蓋前者聽錯。另需區分台語轉華語翻譯、簡繁文字轉換、口音辨識及說話者身分辨識。這些任務應有獨立標籤與成績，不混成一項總準確率。'],
['h','二、目前進度與證據'],
['p','已實讀 tranning/speech_to_text.py、prepare_audio.py、diagnose_stt.py，以及 speech_runs 的 config.json、history.json。既有流程為 16 kHz 單聲道音訊、STFT 對數振幅頻譜、共用 OCR.py 的 CNN＋雙向 LSTM 序列網路，最後以 CTC 訓練、逐幀貪婪解碼。speech_runs 存有約 44 MB 的 best_model.pt；權重檔存在只能證明曾保存模型，不能證明實用辨識能力。'],
['p','commonvoice_zhTW 的 manifest 計有 train 6,654 筆、val 740 筆。本輪只讀清單數量，尚未逐檔核對音訊存在性、授權、說話者、總時長與標註品質。目錄名 zhTW 不能證明全部都是台灣華語，更不能當作閩南語或中國大陸口音的涵蓋證據。現有紀錄未提供這些語言／口音的獨立測試結果，因此目前對兩項新增能力的回答均為「尚未驗證」。'],
['p','歷史 epoch 1：train loss 8.0188、val loss 7.1522、val CER 1.0000；epoch 30：train loss 4.3928、val loss 6.7803、val CER 0.92995，亦為檔內最低 CER。這是舊驗證紀錄，不是本輪重新盲測。CER=(替換＋刪除＋插入)/參考字元數，不能把 1−CER 當成一般意義的辨識準確率；目前結果仍顯示嚴重辨識問題。'],
['h','三、應先修復的訓練流程'],
['p','目前設定 max_frames=200、hop_length=128；以 16 kHz 換算，時間窗約為 1.6 秒（邊界受 STFT 設定影響）。_fit_time_frames 會裁去超出的聲音，卻仍使用整句文字標籤，可能形成音訊中已沒有的答案。應先統計各長度區間，再依完整語句切段、採動態長度批次與長度分桶；不能只把上限放大便宣稱問題已解決。'],
['p','現有字元編碼會略過不在字表中的字，可能讓標籤與原逐字稿不一致。應在訓練前列出未知字元、保留來源、拒絕靜默刪除；字表只由訓練資料建立，再報告 dev/test 的未知比例。input_lengths 目前以補齊後的統一輸出長度提供，需改為每筆有效長度經 CNN 下採樣後的真實長度，並檢查重複標籤所需的 CTC 最短對齊。zero_infinity=True 可讓不可能對齊的無限損失歸零，因此必須另計無效樣本數，不能只觀察平均 loss。'],
['p','雙向 LSTM 需要未來上下文，現版適合停止錄音後辨識，不能標為即時串流。辨識函式目前會套 remove_fillers；應另外保留原始解碼與去語助詞結果，避免刪除台語常用語素或使評分失真。上述都是靜態檢查發現的風險，尚未完成對照實驗，不將它們說成舊模型低成績的唯一原因。'],
['h','四、閩南語與台灣台語的學習設計'],
['p','先以台灣台語為範圍，按資料實際涵蓋的腔調、說話者和生活場景建立分組；不因為訓練過台語就宣稱支援全閩南語、泉州話、漳州話、廈門話或其他變體。第一批可涵蓋問候、購物、數字、時間、地點，以及本專案的電學詞彙，並收錄華語／台語混用的自然句，不只朗讀固定口令。'],
['p','每段音訊至少保留原始逐字稿、標準化逐字稿、語言標籤、說話者代碼、原錄音代碼、起訖時間、授權與標註者。主輸出先依可取得且可核對的標註資源選擇台語漢字或台羅；有可靠配對標註的部分再附另一書寫層，不可把華語翻譯當作台語逐字稿。台羅需固定連字符、音節切分、調號與 Unicode 正規化；本調或表面變調的標記政策要寫清楚，由熟悉台語者核對。沒有足夠標註時，先做單一輸出基準，再研究漢字／音節雙頭或輔助目標，避免同時增加不可靠標籤。'],
['p','資料候選以 NYCU 官方列出的 TAT／TAT-MOE 為優先調查對象，實際取得方式與使用條件仍要逐項確認。官方頁寫有多麥克風通道，多通道重複收音不可當作多倍獨立語句；同一人的同一句在不同麥克風錄音應歸同一資料群。MinSpeech 可作研究參考，但其普通話文字標籤涉及跨語言轉換，不能直接當台語漢字逐字聽寫的等價答案。[S1][S2]'],
['h','五、中國大陸普通話與地方口音'],
['p','目前未完成專門資料盤點、分組訓練或盲測，沒有可報告的中國大陸口音成績。「中國大陸地區」在本報告暫解讀為普通話及帶地方口音的普通話；粵語、吳語、客家語等獨立語言／方言語音不能只靠普通話資料就宣稱支援。若後續要求辨識地區來源，應另立語言／口音分類任務，且不可把口音推測當作出生地或身分的確定判斷。'],
['p','AISHELL-1 官方資源可作普通話基準候選：頁面記載邀請來自不同中國口音區的 400 人，在安靜室內錄音並降採樣至 16 kHz，另有語音與逐字稿。這不代表每種地方口音都有足夠樣本，也不代表自然對話、戶外噪音或台灣華語泛化已經驗證。頁面列 Apache 2.0，同時有學術使用敘述；正式採用時仍應保存下載版本、完整條款與來源記錄。本輪未下載該資料。[S3]'],
['p','顯示繁體中文可在聲學解碼後另做文字轉換，但需保留原稿與轉換稿，對一對多用字與專有名詞做人工抽查。簡體轉成繁體只解決字形，不能解決聲母韻母、兒化、語速或詞彙差異。測試分為台灣華語、普通話、指定地方口音、華語台語混用與噪音情境，逐組報告樣本數及錯誤；資料不足的地區明列「無法估計」，不合併成全地區支援率。'],
['h','六、資料隔離、訓練路線與驗收'],
['p','先依原始錄音、說話者與來源群切分 train/dev/test，再切短片與做增強。同一句的改速、加噪、重採樣與多通道版本不得跨集合。除固定詞彙測試，也要保留新句型、未見說話者及新設備；若同一套朗讀稿跨說話者重複，應另報讀稿重疊程度。低品質或翻譯式標註不可混入忠實聽寫的金標測試集。'],
['p','模型路線先維持既有 CRNN＋CTC，修復標籤與有效長度，建立可信基準；接著以相同資料與預算比較 log-Mel 前端及小型自建 Conformer＋CTC。架構本身不保證改善，也不把外部模型權重納入方案。RTX 4060 8 GB 是設計上限而非已測容量；批次大小、音訊長度與模型參數需經實測。雲端環境及 GPU 尚未就緒，不估計虛構時長；實際執行後以暖機後多個 batch 的速度計算 ETA，保留停止條件與獨立輸出目錄。'],
['p','驗收同時報告字元錯誤率、整句正確率、台羅音節錯誤率（若有可靠音節標籤）、數字／單位精確率、靜音與非語音的亂出字比例，以及推論延遲、RTF、峰值顯存。RTF=辨識耗時/音訊時長，須註明設備、批次、解碼方式及是否包含前處理；長度、噪音和語言分組均保留原始輸出。理解測試另在固定人工逐字稿上測意圖／槽位，再與完整語音管線對照，分清聽錯與想錯。'],
['p','過擬合判斷需在固定無增強 train 評估集與 dev 上比較 CER、loss 及趨勢；若 train 改善但 dev 惡化，先查說話者差異、標註規則與資料洩漏，再調正則化。兩者均差可能是標籤對不齊、空白輸出崩塌、詞表缺字、容量或更新不足，不僅是 epochs 太少。測試集不作選模型依據，詞庫及語言模型補字的對照成績另列，不能讓工具修正冒充聲學模型學會。'],
['h','七、三方協作紀錄與執行邊界'],
['p','COLLAB_PLACEHOLDER'],
['p','下一步應先完成資料稽核報告、修復提案與小批人工金標測試，確認合法可用音訊後再提出具體訓練輪次。先做少量乾淨錄音的可學習性實驗，再擴大到多說話者與混語；小樣本記住只代表管線有學習能力。既有 Linear、電路理論與 SVG 任務不因本報告而標示完成。本輪無新增訓練、無模型切換，也未執行關機。'],
['h','八、資料來源與可追溯證據'],
['p','[L1] 專案原始碼：tranning/speech_to_text.py（音訊前處理、CTC、推論）；tranning/OCR.py（共用 CRNN）；tranning/prepare_audio.py（轉檔與切段）；tranning/diagnose_stt.py（診斷工具）。查閱日期：2026-11-03。上述相對路徑均位於 C:/Users/roy/Documents/GitHub/AI-module/。'],
['p','[L2] 本機紀錄：tranning/speech_runs/config.json、history.json、best_model.pt；tranning/commonvoice_zhTW/train.json、val.json。僅作現況證據，非新測試或資料授權證明。既有規劃：team-run/speech-recognition/PLAN.md。'],
['p','[S1] NYCU Speech AI Research Center，Corpora（官方資料索引與授權連結）。https://sites.google.com/nycu.edu.tw/sarc/tw-nlai/corpora 。查閱日期：2026-11-03。僅依官方頁辨識 TAT 等候選，不將收錄時數直接換算為獨立可用樣本或已取得資料。'],
['p','[S2] Lin 等，MinSpeech: A Corpus of Southern Min Dialect for Automatic Speech Recognition，INTERSPEECH 2024，官方專案頁 https://minspeech.github.io/ 。查閱日期：2026-11-03。頁面說明提供連結與標註而非音訊，標註用普通話文字，CC BY-NC-SA 4.0 不涵蓋原影音權利的自動轉授。故僅列研究候選，不自動下載，不直接當台語逐字稿。'],
['p','[S3] OpenSLR SLR33，AISHELL-1 官方資源頁。https://www.openslr.org/33/ 。查閱日期：2026-11-03。僅用其錄音條件與發布資訊說明候選資料特性，不引用他人模型成績作本專案能力證明。'],
];
let collab=fs.existsSync(__dirname+'/collaboration.txt')?fs.readFileSync(__dirname+'/collaboration.txt','utf8'):'Claude 與 Grok 已收到本輪語音方案任務，尚在回覆；正式互審結果待記錄。';
let children=blocks.map(([k,t])=>new Paragraph({heading:k==='title'?HeadingLevel.TITLE:k==='h'?HeadingLevel.HEADING_1:undefined,alignment:k==='title'?AlignmentType.CENTER:AlignmentType.JUSTIFIED,spacing:{after:k==='h'?90:70,before:k==='h'?160:0,line:260},keepNext:k!=='p',children:[new TextRun({text:t==='COLLAB_PLACEHOLDER'?collab:t,font,size:24,bold:k!=='p'})]}));
const doc=new Document({styles:{default:{document:{run:{font,size:24},paragraph:{spacing:{line:260}}}}},sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1000,bottom:1000,left:1000,right:1000,header:450,footer:450}}},headers:{default:new Header({children:[new Paragraph({children:[new TextRun({text:'AI-module｜語音辨識與方言能力研發報告',font,size:24})]})]})},footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({text:'Codex｜2026-11-03　｜　第 ',font,size:24}),new TextRun({children:[PageNumber.CURRENT],font,size:24}),new TextRun({text:' 頁',font,size:24})]})]})},children}]});
Packer.toBuffer(doc).then(b=>fs.writeFileSync(__dirname+'/(Codex-語音識別與閩南語及中國大陸口音).docx',b));


