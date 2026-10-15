# 2026-09-27 修正完畢：jieba/main.py 多項錯誤

## 對應錯誤
`jieba/main.py`（新檔案，練習用 jieba 斷詞 + TF-IDF 腳本）一次含有多項錯誤：

1. `read_news()` 內用到 `title_files.append(file)`，但 `title_files` 從未定義 → `NameError`
2. `jieba_TFIDF()` 內 `stopwordslist(stopwordslist('C:\\Users\\wmnl\\Desktop\\stopwords.txt'))`
   雙重巢狀呼叫，且路徑是別人（wmnl）電腦上的桌面路徑，不是本機路徑
3. `text.append(str(words))` 邏輯錯誤：每輪迴圈都把整個 `words` 列表塞進去，
   而不是當輪文件的斷詞結果
4. `vectorizer.get_feature_names()`：sklearn 1.9.0 已移除該方法（改名
   `get_feature_names_out()`），呼叫會 `AttributeError`
5. `TfidTransformer`：拼字錯誤（少一個 `f`），應為 `TfidfTransformer`，且未 import
   `CountVectorizer`/`TfidfTransformer`
6. `title_file[i]`：少了 `s`，應為 `title_files[i]` → `NameError`
7. `read_news()` 用內建型別名 `str` 當區域變數名稱，蓋掉內建函式
8. Pylance/Pyright 靜態檢查錯誤：`weight = tfidf.toarray()` 報
   `Cannot access attribute "toarray" for class "ndarray[...]"`——
   `TfidfTransformer.fit_transform()` 的型別標註在型別檢查器眼中被推斷成
   `ndarray`（純陣列，沒有 `.toarray()`），實際執行時期回傳的是 scipy 的
   `csr_matrix`（稀疏矩陣，才有 `.toarray()`），只是型別標註沒對齊，導致
   靜態分析誤報。

## 根本原因
這支腳本像是從舊版教學／範例複製貼上，未依專案環境與現裝套件版本檢查過：
變數命名前後不一致（`title_files` vs `title_file`）、缺 import、sklearn 版本
落差（`get_feature_names()` 是舊版 API）、以及 sklearn 的型別標註對稀疏矩陣
回傳值標得不夠精確，導致 Pyright 誤判成 `ndarray`。

## 修正內容
- 補上 `from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer`
- 檔案頂部新增 `title_files = []`
- `read_news()` 改用 `with open(...) as fp: corpus.append(fp.read())`，不再用
  `str` 當變數名稱
- `stopwordslist()` 呼叫改成單次呼叫，路徑改成專案相對路徑
  `os.path.join(os.path.dirname(__file__), "stopwords.txt")`
- `text.append(" ".join(seg_list))` 取代錯誤的 `text.append(str(words))`
- `get_feature_names()` → `get_feature_names_out()`
- `TfidTransformer` → `TfidfTransformer`
- `title_file[i]` → `title_files[i]`
- 針對 Pyright 的 `.toarray()` 誤報：新增 `from scipy.sparse import csr_matrix`，
  並在 `tfidf = transformer.fit_transform(X)` 後加一行
  `assert isinstance(tfidf, csr_matrix)`。這是真正會在執行期驗證的型別窄化
  （narrowing），不是單純加 `# type: ignore` 蓋掉警告——如果 sklearn 未來改回傳
  純陣列，這裡會直接 assert 失敗而不是靜默吃下錯誤資料。

## 驗證
`python jieba/main.py` 執行成功，模組層級的 jieba 斷詞示範程式碼正常印出結果，
無例外拋出。

## 留下的獨立問題（跟本次修正範圍無關，未處理）
`read_news()` 需要 `C:\news` 資料夾、`jieba_TFIDF()` 需要專案內
`jieba/stopwords.txt` 停用詞清單，這兩份都是執行環境需要的實際資料檔案，
本機目前都不存在，不是程式碼邏輯錯誤，留給你之後自行補上對應檔案再測試
這兩個函式的實際執行結果。
