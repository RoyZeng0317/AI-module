# 資料分析：chat_eval_cases.json

- 筆數：24　欄位：8　完全重複列：0

| 欄位 | 型別 | 缺值 | 唯一值 | 摘要 |
|---|---|---|---|---|
| id | text | 0（0%） | 24 | 長度 8～17，平均 12.3 字 |
| prompt | text | 0（0%） | 24 | 長度 9～81，平均 34.8 字 |
| language | categorical | 0（0%） | 2 | zh-Hant(12)、en(12) |
| expected_intent | text | 0（0%） | 21 | 長度 11～27，平均 18.4 字 |
| review_criteria | nested | 0（0%） | 24 | 長度 27～161，平均 68.2 字 |
| requires_network | categorical | 21（88%） | 1 | True(3) |
| history | nested | 21（88%） | 3 | 長度 28～107，平均 61.7 字 |
| context_required | categorical | 21（88%） | 1 | True(3) |

## 提醒
- 欄位「id」每列都不同，疑似 ID，不要當特徵（容易背答案）
- 欄位「requires_network」缺值 88%，建議補值或排除
- 欄位「requires_network」只有一種值，對模型沒有資訊
- 欄位「history」缺值 88%，建議補值或排除
- 欄位「context_required」缺值 88%，建議補值或排除
- 欄位「context_required」只有一種值，對模型沒有資訊
- 沒有偵測到目標欄位；要看類別分佈或分層拆分請用 --target 指定

以上數字由程式直接計算（data_analysis.py），不是模型生成。
