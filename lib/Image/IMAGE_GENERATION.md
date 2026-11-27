# 圖形文字生圖：Python第一階段

已完成教材、模型入口及獲准的首輪訓練；**候選模型仍未通過品質門檻**。原始輸出已有基本輪廓，但藍色偏灰、邊緣模糊／格紋，嚴格已見0/24、dev0/6。詳見[首輪結果](../data/imagegen_shapes/PILOT_V1_RESULT.md)。先做固定中文提示的單一實心圖形，之後再延伸本地網頁及桌面GUI。

## 資料與程式

- `data/imagegen_shapes/v1/`：360張64×64合成教材，train240／dev60／test60。紅、藍、綠、黃×圓形、正方形、三角形，整組切分。教材是Pillow繪製，不能當作模型生圖成果。
- `lib/img.py`：沿用自建CVAE，不使用外部模型或API。Python介面 `generate(prompts, out_dir=..., save_to=..., seed=42)` 可供後續網頁及GUI共用。
- `tranning/imagegen/shapes_data.py`：版本化教材建立工具。
- `tranning/imagegen/shapes_score.py`：從原始PNG的像素判斷顏色及形狀，分集合／組合報告；空白、模糊與無法辨識均算失敗。狹窄幾何評分仍需人工看圖。

使用現有 `.venv` 的PyTorch、torchvision、Pillow與NumPy；NumPy用於此生圖評分器，既有卷積練習仍保留原本的純for迴圈、不使用NumPy規定。算力上限仍為RTX4060 8GB。

## 目前可執行

在專案根目錄查看參數：

```powershell
.venv/Scripts/python.exe lib/img.py --help
```

本輪已核准並執行完成，原完整命令與停止條件見[首輪提案](../team-run/imagegen-shapes/PILOT_PROPOSAL.md)。依[待辦#43](../to_do_list.md)每輪訓練前確認，這裡不以工程測試代替核准或品質驗收。

## 使用首輪候選生成（效果未驗收）

```powershell
.venv/Scripts/python.exe lib/img.py --generate "紅色圓形" --seed 42 --out-dir lib/img_runs/t2i_shapes/pilot_v1
```

每次生成另建UUID目錄，保存原始PNG及 `samples.json`（提示詞、seed、模型設定、checkpoint SHA256與批次順序）。無checkpoint時只提示尚未訓練，不用規則繪圖頂替。

同prompt、同seed、同環境及同批次排列可重現；不同prompt要比較相同latent時，每次各生成一張並用相同seed。模型目前是字元平均池化，不理解一般語序，也不保證任意中文提示均有可用效果。

```powershell
.venv/Scripts/python.exe -m tranning.imagegen.shapes_score "實際生成目錄/samples.json" --output "實際生成目錄/score.json"
```

上例須換成真實路徑；評分輸出拒絕覆寫。dev與test分開評估，先看dev，不依test反覆調參。訓練預算採批次邊界檢查，不能中斷正在執行的kernel。完整證據與限制保存在 `team-run/imagegen-shapes/`。

## v2候選選項（已完成核准單輪，品質未過）

`--decoder-kind resize_conv --loss-kind foreground_l1 --foreground-weight 4`明確選用v2；預設仍為v1的`transpose`及`bce`。新run保存架構版本與loss參數；舊config缺新欄位時按v1載入，未知版本／不合權重直接拒絕。v2保持原始sigmoid RGB輸出，不以調色、閾值或模板代替模型生成。

v2 L1用教材相對白底的差值作連續前景權重，逐張平均權重正規化，再以pixel sum per batch相加；KL仍為latent sum per batch。本輪已有部分視覺改善但未通過原門檻；架構與loss同時變更，尚無單項因果驗證。原核准命令與停止條件由[新候選工作紀錄](../team-run/imagegen-shapes-v2/BUILD_MODEL.md)及同目錄提案說明。

文字生圖訓練會保存`run_summary.json`：實際訓練／整輪秒數、更新次數、停止原因、裝置與PyTorch CUDA allocator峰值allocated/reserved bytes。CPU欄位為null；這不是整張GPU總使用量，時間上限仍為合作式檢查，不能硬中斷正在執行的kernel。本輪實際資料見下節及結果報告。

## 第二輪實跑狀態（2026-10-30）

v2已個別核准並完成111epochs／1665updates，early stop後保留best81，候選目錄lib/img_runs/t2i_shapes/pilot_v2。可用同一generate介面／CLI指定該out-dir，但品質未驗收，嚴格seen0/24、dev0/6；confirm/test未開。藍色與格紋目視改善，邊界混色及未見組合形狀干擾仍需處理。詳見[第二輪結果](../data/imagegen_shapes/PILOT_V2_RESULT.md)。
