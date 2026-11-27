# 獨立人物標註與characters.pt

## 單張圖片匯入識別

獨立入口沿用 `tranning/img_mnist_predict.py` 的路徑設定、前處理、get_model、main及命令列區段；MNIST原檔保留。人物仍使用YOLO彩色輸入，不套用MNIST灰階／28x28處理，不會自動訓練。

```powershell
# 開啟Tkinter主視窗，按「匯入圖片」選圖
.\.venv\Scripts\python.exe tranning/img_character_predict.py
# 也可直接指定圖片
.\.venv\Scripts\python.exe tranning/img_character_predict.py "你的照片.png" --conf 0.25
```

可在新程式頂端將 `test_img_dir = None` 改為 `project_dir / '你的資料夾/照片.jpg'`；如此不帶參數執行就自動在視窗載入該圖。模型固定讀取 `tranning/yolo_runs/custom/weights/characters.pt`，缺檔或空檔會顯示錯誤，不換成物品模型。

預設Tkinter視窗有匯入圖片、重新辨識、門檻欄位、等比例圖片預覽，並在圖上繪製繁體中文姓名／代號、框與分數；下方也列出結果。修改門檻後按「重新辨識」，沒有框時顯示「未偵測到人物」。辨識在背景執行，關閉會等待背景工作結束。

若仍需要原本只輸出文字的方式，使用 `python tranning/img_character_predict.py "照片.jpg" --no-gui`。模型結果不是已驗證的身份，也不會自動將陌生人可靠地標成未知。

2026-10-26更新：目前人物權重已完成第一輪訓練，並依30張驗證圖比較選用該run的last.pt發布為characters.pt；物品模型沒有改動。人物視窗可設定「偵測門檻」，預設0.25；若改門檻，請先停止鏡頭後再啟動。這次門檻0.25正確配對24/31框、誤報1框，不等於所有場景或人物身份的準確率。

視窗下方新增「人物發布權重」best／last選擇，目前預設last是本輪比較結果，不代表每一輪last必然較好。發布後會清除模型快取；外部更新characters.pt後，已開啟的程式仍需關閉重開。

在專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe -m lib.yolo.character_labeler
```

1. 預設類別為「人物1」，可按「更名類別」改成姓名或代號，再「新增類別」加入其他人。
2. 匯入照片或從攝影機擷取，選對類別後人工框選。若要學臉部特徵，請一致框選臉部；若選擇全身，則一致框選全身。不要在同一份資料混用兩種框法。
3. 儲存標註，按不同拍攝場景分train／val；畫面中所有要學的目標都需標出，避免漏標。
4. 選用本地起始YOLO權重並按「開始微調」。你建立的空白characters.pt不能作為起始權重；預設沿用專案已有yolo11n.pt。
5. 訓練成功後，程式將「人物發布權重」所選best.pt或last.pt複製到 `tranning/yolo_runs/custom/weights/characters.pt`，並自動選取供本視窗即時偵測。若原檔存在，先備份至同目錄characters_backup下的獨立資料夾。
6. 按「使用自訂模型即時偵測」。下次打開人物工具，非空characters.pt會自動填入偵測權重；真正載入時仍會檢查模型類別與人物資料集一致。

人物資料在 `data/character_detection/`；訓練run在 `tranning/character_detection_runs/`。照片、類別與模型不會寫入原物品資料集，也不會改物品攝影機腳本的模型路徑。

最初characters.pt為0bytes，只是佔位檔；之後使用者已完成真實訓練。更換類別或姓名後仍需重新訓練。自動發布只處理權重檔案，不代表辨識品質已驗收，請用沒參與訓練的場景檢查。

這個流程是YOLO人物類別標記，不是完整的人臉身份驗證。人物姓名類別可能誤認，也可能學到衣服與背景；沒有實作未知人物拒絕、活體辨識或換衣服後仍可靠認人的保證。物件追蹤編號也不等同姓名。
