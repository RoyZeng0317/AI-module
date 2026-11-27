# 第二輪結果（2026-10-30）：有視覺改善，品質門檻未通過

使用者明確選擇「啟動第二輪訓練」後，先重新執行preflight確認train/dev300張教材及固定來源，僅執行提案一次。沒有改教材、評分門檻或原v1權重，沒有追加輪次。

## 訓練與資源

- 自建v2 CVAE：ResizeCondDecoder、零轉置卷積，1,744,099參數；nearest上採樣＋3×3卷積，原始sigmoid RGB輸出。
- foreground-aware L1＋KL，beta1、foreground_weight4；batch16、64px、latent32、cond64、seed42、lr0.001，從零初始化。
- 核准上限180epochs／2700updates／600秒合作式預算，patience30。
- 實際第111epoch因dev損失連續30輪未改善而停止，共1665次更新；最佳候選為第81epoch（train192.9866、dev359.5690）。結束時train185.0419、dev382.6341。不同loss不能與v1的BCE數字直接比較品質。
- 正常exit0，沒有OOM／NaN。run_summary.json：training50.704秒、整輪50.889秒；裝置NVIDIA GeForce RTX4060。
- 本程序PyTorch allocator峰值：allocated83,829,760bytes、reserved109,051,904bytes，非整張GPU總用量，也不包含所有driver/context開銷。
- 候選權重：lib/img_runs/t2i_shapes/pilot_v2/model.pt，6,998,901bytes；SHA256 `72dfdd32c76713d255d2ee42f92ba0fecf21b23ccecfacfaf3040a5aa540fa08`。

## 凍結評分器的原始推論

正式Python評估器依原train/dev的10種prompt，每個單獨以seed42／43／44生成，共30張原始PNG及來源metadata，保存於output/imagegen-shapes-v2/dev。換prompt時同seed使用相同latent起點；沒有推論後調色、palette snapping、二值化或Pillow重畫。

| 集合 | 張數 | 嚴格通過 | 每組結果 |
|---|---:|---:|---|
| 已見train組合 | 24 | 0/24 | 8組皆0/3 |
| 未見dev組合 | 6 | 0/6 | 2組皆0/3 |

30張均被原評分器拒絕：mixed color24、off palette or blurred6。所有拒絕均計入失敗，不剔除難例。`gate_passed=false`，因此confirm及test都沒有開啟，也沒有產出這兩階段目錄。

目視比較同seed、同prompt的v1與v2：藍色由灰轉為較接近藍色，格紋明顯減少，已見圖形輪廓較乾淨；但邊界混色仍在，未見的紅色三角形出現變形及異色邊，藍色正方形亦有異色角落。這是定性改善，尚不符合固定模板逐組80%的事先門檻。

原始聯絡表為output/imagegen-shapes-v2/dev/contact_sheet.png；比較圖output/imagegen-shapes-v2/v1_v2_comparison.png僅排版與最近鄰放大原始PNG，非後製修正。

## 獨立核對及狀態

PILOT_V2_REVIEW.md獨立核對30張原圖與checkpoint/config來源，重算分數完全一致；冷載入重現dev紅三角seed42逐byte相同，直接decoder tensor轉PNG亦一致。原v1與train/dev/scorer固定SHA保持不變。

工程執行證據通過，品質仍未通過，不提升正式預設模型。下一步需診斷前景顏色一致性及未見組合的形狀／顏色干擾，再提出可審查的新候選；每輪更新權重仍依#43個別確認。網頁與桌面GUI尚未擴建。
