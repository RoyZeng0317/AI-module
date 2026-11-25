# 08｜OCR 光學字元辨識

[← 目錄](README.md) ｜ [下一章：總整理 →](09_總整理與測驗.md)

## 核心概念

**OCR（Optical Character Recognition）負責辨識圖片或掃描文件中的文字。**

```text
PCB 上的晶片照片
      ↓
OCR 辨識「LM358」
      ↓
RAG 搜尋 LM358 Datasheet
      ↓
LLM 解釋腳位、用途、注意事項
```

OCR 不等於一般影像辨識：辨識「這是一隻貓」屬於影像理解；辨識照片上的 `LM358` 字樣屬於 OCR。

## Python：Tesseract OCR

安裝 Python 套件：

```powershell
pip install pillow pytesseract
```

**還必須安裝 Tesseract OCR 主程式**並加入 PATH；Windows 可於安裝後指定執行檔位置。若需繁體中文辨識，還要安裝 `chi_tra` 語言資料。

```python
from PIL import Image
import pytesseract

# Windows 若找不到 tesseract.exe，可取消下行註解並調整路徑：
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = Image.open('sample.png')  # 請自行放入要辨識的圖片
text = pytesseract.image_to_string(image, lang='eng')
print(text)
```

若已安裝繁體中文語言包，可使用 `lang='chi_tra+eng'`。

## 常見 OCR 錯誤

- `0` 和 `O`、`1` 和 `I` 容易混淆。
- 圖片模糊、反光、字體太小會影響準確率。
- 晶片型號辨識後，應與 datasheet 或其他可靠來源交叉驗證。

## 延伸

[RAG](07_RAG.md) ｜ [CNN](03_CNN.md)
