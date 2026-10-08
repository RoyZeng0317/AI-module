# Sinco 過擬合：Codex → Claude Code 交接

本次以 Claude 在 to_do_list.md #44 的防過擬合實作為基礎。此環境沒有 Claude Code CLI，尚未與即時 Claude Code 工作階段直接通訊。使用者可以把此文件交給本機 Claude Code 接續。

## 已確認的現況

- data/pairs.json：742 筆，742 個不同的正規化 prompt。
- v2 最佳 epoch 7：train_loss 2.4471、val_loss 3.8526；最後 epoch 17：1.1125／3.9823。
- v3 最佳 epoch 4：train_loss 3.3215、val_loss 4.1382；最後 epoch 14：1.5855／4.1926。
- 驗證 loss 回升而訓練 loss 持續下降，支持既有過擬合診斷；兩次用了不同基底與隨機切分，不能僅凭最佳 loss 判定版本優劣。
- data/corpus_pretrain_v2.txt 中逐字搜尋得到全部 742 筆 prompt 與全部 742 筆 reply。基於此 corpus 的預訓練／rehearsal 不可視為與 pairs 驗證資料隔離。文字包含不等於模型必然記住，但不符合未見資料評估條件。
- GitHub 未提供 gpt_pretrain_runs_v2/model.pt 或 checkpoint/model.pt，雲端無法重跑真實微調。

## 已完成的程式修正

tranning/transformer_chat.py：

1. finetune 的 --split-seed 預設 42，依 strip + casefold 的 prompt 分組切分，同樣資料即使重排也會選到同樣的驗證 prompt。不同回覆但相同 prompt 不會跨集合。這不會自動辨識語意近似句，擴增資料仍須人工依來源／情境分組。
2. 空驗證集、單一 prompt 資料及顯式 train/val prompt 重疊直接報錯，避免拿訓練集當驗證集。這是刻意收緊既有介面。
3. loss 按非 PAD 的目標 token 數加權；最後小 batch 不再與完整 batch 等權。只有 PAD 的 batch 跳過，整個集合無可評分 token 時報錯。
4. 新增 train_eval_loss：同一 epoch 結束後，eval 模式、未加錯字、未混 rehearsal、未加 label smoothing 的訓練 pairs CE。過擬合警告用它與 val_loss 比較。train_loss 仍保留為實際訓練目標，開 rehearsal 時是混合目標，不能直接拿它衡量對話泛化落差。
5. config.json 紀錄切分 seed、train/val prompts、筆數與 loss reduction。seed 只固定切分，未固定 GPU 或訓練隨機性。

注意：新增乾淨訓練集評估會增加每 epoch 的 forward 時間。既有 history 是 batch mean，新版是 token mean，不能直接沿用舊數字比較改善。

## 本機 Claude Code 接續

先審閱補丁，確認本機未提交的修改沒有衝突。保留 #44 的 freezing／label smoothing／typo augmentation 機制。新增資料時不要再把驗證／測試對話混入預訓練或 rehearsal。準備全新人工寫的測試對話，涵蓋中英文日常、否定、情緒及未見措辭，固定後不要拿來調參。

使用相同預訓練權重、同樣 pairs、seed、dropout、weight_decay、lr、batch_size，重跑基準與防過擬合設定。第一輪暫不使用 --lm-corpus，因為現有 corpus 含驗證對話；評估 rehearsal 時要換成獨立語料。以下指令只供本機實驗準備，此次未執行真實訓練。依既有專案紀錄，啟動長訓練前取得使用者授權；不要自動切換產線 checkpoint。

在專案根目錄執行（需本機權重存在）：

```bash
python tranning/transformer_chat.py finetune --data data/pairs.json --pretrain-dir tranning/gpt_pretrain_runs_v2 --out-dir tranning/gpt_chat_runs_baseline_fixed --epochs 30 --batch-size 8 --lr 0.0001 --dropout 0.2 --weight-decay 0.05 --patience 5 --checkpoint-every 1 --gpu-mem-fraction 0.85 --split-seed 42 --no-freeze-embeddings --freeze-layers 0 --label-smoothing 0 --typo-noise-prob 0
python tranning/transformer_chat.py finetune --data data/pairs.json --pretrain-dir tranning/gpt_pretrain_runs_v2 --out-dir tranning/gpt_chat_runs_regularized_fixed --epochs 30 --batch-size 8 --lr 0.0001 --dropout 0.2 --weight-decay 0.05 --patience 5 --checkpoint-every 1 --gpu-mem-fraction 0.85 --split-seed 42 --freeze-embeddings --freeze-layers 4 --label-smoothing 0.1 --typo-noise-prob 0.3
```

使用 gpt_pretrain_runs_v2 的最終輸出（程式會恢復最佳權重），而非默認挑 checkpoint/ 的最近權重。若本機僅有 checkpoint/，兩次實驗必須一致指向它並紀錄 epoch。

比對最佳 val_loss、同 epoch 的 train_eval_loss、驗證反彈幅度及全新測試句的實際回答。先比較 baseline 與整套既有修正，再視結果逐一移除設定做消融；不要一次跑全部组合。742 筆的資料量仍有限，機制通過測試不代表真實泛化已改善。即使固定微調驗證集，既有 pretrain 看過其文字的限制仍存在；全新測試集才是外部驗證。

## 驗證

CPU 合成資料／回歸測試，沒有真實模型訓練：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tranning/test_transformer_chat.py tranning/test_reward_model.py tranning/test_train_grpo.py
```

結果：28 passed。新增測試驗證切分與順序／全域 RNG 無關、prompt 不跨集合、拒絕假驗證、不同 batch size 的 loss 與獨立 token CE 一致，以及微調保存乾淨 loss 與切分紀錄。git diff --check 通過。
