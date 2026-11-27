# 首輪實際訓練與評估（2026-10-28）

使用者於本輪明確核准啟動。以v2資料、原正式 `gpt_chat_runs` 為基底，完成3 epochs、66次optimizer更新，另存 `tranning/gpt_electricity_open_runs/pilot_v1`，沒有切換正式聊天模型。

## 訓練

| Epoch | Train loss | Dev loss |
|---|---:|---:|
| 1 | 4.2818 | 5.9756 |
| 2 | 4.0512 | 5.6536 |
| 3 | 3.7697 | 5.4044 |

保存最佳dev loss的第3 epoch。44筆訓練＝22筆電學＋22筆聊天重播；batch2、lr1e-5、僅最後block與ln_f可更新。14個tensor變動，embedding／共用head及前7層不變，全部權重有限值。各epoch有可能過度擬合的loss差距警示；不能僅由此判定偏題原因。

候選SHA256：`ea38c91f5bed483affbea0cb3191b9089cc338512b165738e11089e6c013eb02`。

## 原始生成品質

直接呼叫GPT，繞過公式工具與聊天規則。基底與候選使用相同問題、每題相同seed及解碼設定（CPU、temperature .7、top_k30、top_p .85、repetition_penalty1.3、最多180 tokens）。全部原始回覆先落地，之後才讀評分鍵；由獨立Reviewer人工逐題核對，不使用「包含正確數字」的子字串給分。

| 評估 | 基底 | 候選 |
|---|---:|---:|
| Dev 10題 | 0/10 | 0/10 |
| 首次test 10題 | 未作答 | 0/10 |
| 已見train抽樣5題（僅診斷） | 0/5 | 0/5 |

候選test分項為數值0/7、資訊不足／短路0/2、節點／網目0/1，全數偏題。100Ω／20V原題、200Ω／40V改寫與固定mesh的3題回歸也未答對。這不是只有格式缺失，而是尚未產生有效的物理解答。

四個一般聊天樣本前後逐字相同，但原先陪聊句本身不流暢；這只表示此次抽樣未新增退化，不能保證一般聊天能力全部保留。

## 決定與後續

**工程執行通過，生成品質未達標。** 保留候選與紀錄供診斷，不升為預設，不自行續訓。loss下降不等於學會電學；即時公式答案也不能計入GPT成績。

本次test已首次使用並開封，之後只能作已知回歸資料，不能再稱新的盲測。下一輪需另訂未見題與實驗設定、逐輪核准，不能靠反覆調到這10題答對來宣稱泛化。

證據：`output/electricity-open/pilot_v1_training.log`、`output/electricity-open/pilot_v1_evaluation/`、`team-run/electricity-open/PILOT_V1_REVIEW.md`。正式基底與五份受保護檔案指紋保持不變。
