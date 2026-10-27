# Transformer 現況獨立審查

本次只做靜態讀取與JSON統計；未載入模型、訓練、修改資料／權重、讀取.env或呼叫API，也未占GPU。未執行原始回覆抽樣，該項由父代理處理。

## 可確證結論

1. **一般Transformer分支沒有收到對話history。[High]** GUI其實有保留與傳遞：`lib/components/conversation.py:122`建立history，`:169`把history交給smart_reply_traced，`:193`追加，`:194`保留20輪，`:198`及`:205`寫入持久對話紀錄。但`tranning/chats.py:712`實際只呼叫`transformer_chat.reply(message)`，沒有轉交history。因此「GUI有顯示／保存對話」與「本機模型能讀到前文」是兩件事。`command.py:418`與`:489`的恢復history也不會改變此呼叫契約。
2. **reply只編碼當前message，沒有獨立system／history參數。[High]** `tranning/transformer_chat.py:781`函式介面及`:795`顯示輸入為`tokenizer.encode(message)+[SEP]`，截取最後block_size tokens；`:797`開始生成。目前`gpt_chat_runs/config.json:2`設定block_size512，長訊息開頭可能被截掉；`transformer_chat.py:348`生成時也只保留最後block_size。這能解釋未傳前文的失憶與長輸入前段要求遺失風險，但不是所有「不遵守要求」的唯一原因。
3. **ChatSFTDataset是單一prompt/reply契約，沒有自動構造多輪角色格式。[High]** `transformer_chat.py:434`逐pair處理，`:438`編碼prompt，`:439`拼接prompt+SEP+reply+EOS，`:455`遮罩prompt loss。`finetune`在`:701`／`:708`使用這個Dataset。若某pair的prompt本身已人工串入多輪文字，Dataset仍能當一般文字讀入；不能從這個類別單獨斷言所有實際訓練資料完全沒有多輪文字。能確定的是沒有從history/messages自動產生多輪結構，也沒有system角色loss／權限區分。
4. **一般GUI送出沒有自動注入持久memory或system規則。[High]** `conversation.py:214`將user_text原樣交給ask；`command.py:291`至`:325`的/memory主要新增／刪除／顯示清單，沒有把清單加入一般Transformer輸入。搜尋tranning/lib實作後，實際把list_memories組成system prompt的是`lib/NVIDIA.py:37`、`:91`至`:97`，只屬手動nvidia模式；本次沒有呼叫它。
5. **header_tag="system"只是畫面tag，並非模型system角色。[High]** `/rules`等Markdown指令在`command.py:243`讀檔、`:244`把全文當本次model_input呼叫ask，故該次有機會送到Transformer，但沒有永久建立system狀態，也因未設record_as而不追加普通history。`conversation.py:226`開檔亦同。@附檔在`command.py:270`串到當前訊息，是該次有效文字輸入，並非「全部文件完全不會進模型」。後續回合仍不會自動帶回。GUI/模型應區分畫面系統訊息、當次文字提示與持久高優先規則。

## best／last核對

- [High] `tranning/gpt_chat_runs/history.json`共41列：最低val_loss在epoch26，train_loss=.5156929499622601、val_loss=2.129414528608322（history.json:128至130）；最後epoch41的train_loss=.24652785863449325、val_loss=2.1556280702352524（:203至205）。只能說這份紀錄後段訓練loss續降、驗證loss未再優於epoch26；不是直接的語意品質分數。
- [High] `_run_epochs`在`transformer_chat.py:575`比較最低val_loss，`:577`複製最佳state，`:585`至`:586`結束後載回best_state；finetune在`:733`呼叫它，`:743`才保存model.pt。因此正常完成此路徑的最終檔應是該次best驗證權重，不是單純last epoch權重。`:571`至`:572`的中途checkpoint才明確保存當下權重，需和最終model.pt區分。
- [Medium] 不能只靠history.json證明現在磁碟model.pt一定來自epoch26：歷史檔沒有權重epoch/hash綁定，檔案可能曾被另外替換；本次未比對state與逐epoch檔。亦不能由最後一列指認目前推論一定使用epoch41。`_load_gpt`在`:755`至`:757`還會快取已載入模型；程序未重啟時可能繼續使用先前記憶體版本。

## 父代理報告應避免的過度推論

- [High] 可以明確寫「一般本機分支丟失history，與跨輪無法記得相符」；不可擴張成「Transformer天生不能記憶」或「模型絕對無法理解當前訊息內的上下文」。
- [High] 可以寫「一般分支未自動注入memory／持久system規則」；不可寫「memory檔沒儲存」或「所有模式都忽略規則」。nvidia分支有注入，當次/rules與@附檔也會成為文字輸入。
- [High] `chats.py:648`至`:651`仍以「字元級模型」解釋不帶history，但`:693`以後已改成Transformer，註解理由部分過時；判斷應依`:712`實際呼叫，而非拿舊註解當目前模型架構證據。
- [High] 除history外，`chats.py:662`工具路由、`:678`程式檢索、`:704`角色、`:709`精確句型及`:713`輸出檢查都可能改變使用者看到的回覆。應將raw Transformer與完整路由抽樣分開，不把精確固定回覆說成模型能力，亦不把路由結果誤當原始模型輸出。
- [Medium] 單輪訓練介面、有限context及取樣預設（`:782` temperature=.7等）可形成風險假說，但遵循指令品質還需原始輸出、訓練資料與驗收案例支持；靜態讀碼不能量化模型能力或證明僅改傳參就能達標。本次不實作改造。

## 本次證據邊界

[High] 已獨立讀取實際呼叫鏈、GUI儲存及恢復、memory指令與NVIDIA注入來源、SFT建構、best恢復與保存路徑，並以JSON統計核對41epoch的best／last。未讀訓練pairs全文、未測UI、未驗證checkpoint來源hash、未rawreply取樣，故不對上述未測項作實測主張。
