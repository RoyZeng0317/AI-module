# 07｜RAG 檢索增強生成

[← 目錄](README.md) ｜ [下一章：OCR →](08_OCR.md)

## 核心概念

**RAG（Retrieval-Augmented Generation）不是單一神經網路，而是一套「先檢索相關資料，再讓生成模型參考資料回答」的工作流程。**

```text
使用者問題
   ↓
資料檢索（關鍵字或向量相似度）
   ↓
取得相關片段（Context）
   ↓
問題 + Context → LLM
   ↓
產生有依據的回答
```

## 核心組件

- **Documents**：PDF、網頁、文件、資料庫等知識來源。
- **Chunking**：將長文件分段。
- **Embedding**：把文字轉成向量，方便比較語意相近程度。
- **Retriever**：檢索相關片段。
- **Prompt / Context**：將找到的資訊提供給 LLM。
- **Generation**：由 LLM 根據問題和資料產生回答。

> RAG 不等於 OCR。OCR 讀取圖片中的文字；RAG 負責檢索相關資訊。多模態 RAG 也可以檢索圖片等資料。

## Python：不依賴外部套件的簡易檢索示範

此範例只示範「檢索」與「組合提示」，**沒有真正呼叫 LLM**，也不是語意向量檢索。

```python
import re

knowledge = [
    'LM358 是雙通道運算放大器。',
    'CNN 常用卷積層提取影像特徵。',
    'Max Pooling 會在區域內取最大值。',
]

def tokens(text):
    # 教學用：中文用字元重疊，英文用詞彙重疊
    return set(re.findall(r'[A-Za-z0-9]+|[\u4e00-\u9fff]', text.lower()))

question = 'CNN 如何提取影像特徵？'
query_terms = tokens(question)
ranked = sorted(
    knowledge,
    key=lambda doc: len(query_terms & tokens(doc)),
    reverse=True,
)
context = '\n'.join(ranked[:2])
prompt = f'請只根據以下資料回答；資料不足請說不知道。\n資料：\n{context}\n問題：{question}'
print(prompt)
```

## 真正的語意 RAG 還需要什麼？

1. 使用 Embedding 模型產生文件與問題向量。
2. 以向量資料庫或近似最近鄰搜尋找候選片段。
3. 視需要加入 reranker（重排序）。
4. 將相關片段和來源交給 LLM，並檢查答案是否有根據。

## 常見問題

- 找到錯誤資料，回答仍可能錯。
- 資料分段太碎可能失去上下文；太長又會增加成本。
- RAG 不會自動保證零幻覺，應要求來源並做驗證。

## 延伸

[OCR：從圖片取文字](08_OCR.md) ｜ [Transformer：LLM 常見架構](06_Transformer.md)
