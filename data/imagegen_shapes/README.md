# 簡單圖形教材 v1

本資料由 Pillow 演算法合成，**只是訓練教材，不是模型生圖成果**。未執行訓練。64×64 RGB、白色背景、單一實心物件；位置與尺寸是干擾變化，不是文字控制項。大小24至40像素，左上座標各8至15；圓與方使用外接框，三角頂點朝上。

固定 caption 是顏色加形狀，無空白：顏色 `紅色/藍色/綠色/黃色`；形狀 `圓形/正方形/三角形`。例如 `紅色圓形`、`藍色正方形`。每個顏色與每個形狀都在 train 出現；顏色RGB依序為(220,40,40)、(40,80,220)、(40,170,65)、(235,195,30)。

| 集合 | 完整組合 | 筆數 |
|---|---|---|
| train | 紅圓、紅方、藍圓、藍三角、綠方、綠三角、黃圓、黃方 | 240 |
| dev | 紅三角、藍方 | 60 |
| test | 綠圓、黃三角 | 60 |

每組30張。固定seed42；分組互斥，360張像素SHA256皆唯一。同一組不跨集合；不是隨機逐張切分。`v1/spec.json`保存組合、種子、統計及manifest雜湊，各筆保存PNG與像素雜湊。路徑相對manifest所在目錄，`image/caption`可直接供既有 CaptionedImageDataset；另有color/shape/combo/split/source/box。

train供權重更新；dev供模型選擇。test不可加入訓練或調參，首次正式模型驗收後封存結果；反覆依test修正需另建版本。這裡的教材test圖片已合成並可做完整性檢查，不等於已驗收模型。

建立新版本（拒絕覆寫非空目錄）：

```powershell
.venv/Scripts/python.exe -m tranning.imagegen.shapes_data data/imagegen_shapes/v2 --seed 42 --variants 30
```

對**模型保存的原始PNG**評分，不用教材圖片代替：

```powershell
.venv/Scripts/python.exe -m tranning.imagegen.shapes_score output/your_run/samples.json --output output/your_run/score.json
```

輸入可為manifest list，或含`samples` list的生成紀錄dict。每筆有image相對路徑，target或metadata中的color/shape/split；未提供標籤則只接受完整固定caption/prompt。未知prompt拒絕猜標籤。缺split依本版本固定8/2/2組合映射；combo由目標color/shape計算，若提供錯誤combo或split直接拒絕，空manifest亦拒絕。報表保留seen/unseen總計`groups`，另提供`by_split`及`by_combo`，每筆details含expected/split/combo。dev/test與各組合須個別判讀，不以unseen總分宣稱test通過。後續正式評估仍應分開對dev、test跑。JSON包含逐張判斷與joint accuracy，無法辨識仍計入分母作失敗。

評分器是狹窄的像素幾何量測，不是通用分類模型。要求64px、白背景、前景200至1900像素、近正方外框且不觸邊，98%前景同色且距指定RGB不超過45，背景最多1%明顯偏白；每列前景連續、中心漂移最多1.5px。列寬對方形/圓形/三角形理論曲線平均誤差最多0.065，面積填充比分別0.96–1.01/0.73–0.85/0.46–0.57。旋轉、描邊、多物件、柔和或偏色結果可能被拒絕，必須另外人工看圖。獨立解析式像素正例與空白、雜訊、模糊、錯色錯形反例校準通過，不代表對任意生成圖可靠，也不代表模型已學會。
