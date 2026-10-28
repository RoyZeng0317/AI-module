# 開放式基本電學試驗

首輪已於2026-10-28經使用者核准完成3 epochs／66次更新，但dev與首次test均0/10，**尚未證明模型能力改善，未切換正式模型**。完整結果見[首輪報告](PILOT_V1_RESULT.md)。沿用自建 GPT，不使用雲端模型。候選資料 `v2` 有 train22／dev10／test10，題目只有問題，私有評分鍵另放 `private/`。私有指流程隔離，並非加密。

來源是明確生成的小型資料，並非從使用者 ZIP 抽取。求 I/V/R、千歐姆換算、缺值、零電阻及節點／網目相互改寫均有範例。模板群與完整數值組跨 split 不交疊；但公式與三支路拓撲相同、數值種類很少，不能當成新拓撲或一般電學泛化證據。缺值／短路題要人工評閱。100歐姆20V只作另外回歸題，不進資料集。

先讓學生直接讀題作答，每筆原始輸出即持久化；完整 train 首答保存後，訓練才讀標準回答。訓練採既有 `ChatSFTDataset` 的標準監督微調：問題部分不計 loss，答案部分計 loss；teacher forcing 會把答案前綴作為後續 token 的輸入。這仍然使用答案監督，並非無監督、純獎勵或模型憑空自學；不把完整解答加在使用者問題內。

```powershell
# 已完成，不需重複（拒絕覆寫）
.venv/Scripts/python.exe tranning/electricity_open_training.py prepare
.venv/Scripts/python.exe tranning/electricity_open_training.py attempt-train
# 可重跑；檢查全部 split 的欄位、指紋、群組、UNK、總 token 預算，不訓練
.venv/Scripts/python.exe tranning/electricity_open_training.py preflight
# 下列首輪已核准並完成；輸出已存在，不重跑。下一輪需依#43另行確認。
.venv/Scripts/python.exe tranning/electricity_open_training.py train --epochs 3 --base tranning/gpt_chat_runs --output tranning/gpt_electricity_open_runs/pilot_v1
```

固定 seed43，batch2、lr1e-5、freeze_embeddings=True、freeze_layers=7，僅最後一層與末端正規化可更新。加入最多22筆固定抽樣一般聊天重播（約1:1），dev獨立，test不作 loss／選 checkpoint。CUDA allocator上限75%，不代表整機顯存上限。新輸出路徑不可已存在，不覆寫原 checkpoint。所有 prompt+SEP+reply+EOS 皆檢查不可超過513個 token／不可含 UNK，不裁切答案。有限梯度、權重、history loss 與真正 optimizer 更新次數受檢查；保存最佳 dev loss 權重，仍須訓練後人工／數值評閱未見開放題與聊天回歸。

原始七題基準在 `team-run/electricity-open/raw_gpt_baseline.jsonl`（其中有 train 範例，僅診斷，不能當未見評估）；正式候選訓練題首答在 `v2/train.attempts.jsonl`。初版及中間草稿留於 `draft_initial/`、`v1/`，不是本次訓練來源。v2 使用 Fraction 精確分數計算、不同電流與單位變換結果，跨 split 隔離的是完整數值組，個別數值可以相同。解題示範見 [節點／網目說明](NODAL_MESH_GUIDE.md)。
