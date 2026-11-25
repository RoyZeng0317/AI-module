# 首輪結果：未達品質門檻

2026-10-29，使用者明確選擇「啟動這一輪訓練」，主協調者僅執行已核准的PILOT_PROPOSAL命令一次。沒有追加輪次、改評分門檻或切換正式模型。

## 訓練證據

- 自建TextToImageVAE、2,045,827參數、64px、latent32／cond64、batch16、seed42、lr0.001、beta1。
- train240／dev60；test未交給訓練函式。
- 正常完成60epochs，依每epoch15批計900次更新；沒有stop.json，命令exit0，未觸發OOM／NaN。
- 第一epoch train7137.13／dev6371.48；最佳及最後epoch60 train1400.25／dev1744.05。loss下降不能代替生成品質。
- checkpoint：lib/img_runs/t2i_shapes/pilot_v1/model.pt，8,205,621bytes。
- SHA256：`7ed7696177a9e44dd3089b81f9208d05ef4d37eecea9efdf1c6060b6de7ec0ae`。
- 沒有可靠的整輪牆鐘與峰值顯存紀錄。單次nvidia-smi曾顯示整張GPU1279MiB／8188MiB、41%利用率；不能視為本模型獨占或峰值用量。

## 原始模型推論

`evaluate_pilot.py`只讀train與dev的caption，每個prompt各自以seed42／43／44生成，換prompt時同seed使用相同latent。生成使用模型decoder，沒有用教材或Pillow重畫替換；Pillow只將原始PNG排成預覽。

- 已見8組×3seed＝24張：嚴格joint通過0/24，24張無法評分（mixed color18；off palette or blurred6）。
- dev保留2組×3seed＝6張：嚴格joint通過0/6，6張mixed color。
- test不評估：dev未達門檻，不用封存test作下一輪調參資訊。

原始PNG、逐張samples.json、train/dev分數與來源紀錄在 `output/imagegen-shapes/pilot_v1/`；總覽為 `model_samples.png`。嚴格評分器要求接近教材色與乾淨邊界，0分表示未通過此預定標準，不等於所有輸出都沒有可辨識輪廓。

目視可見圓／方／三角輪廓隨prompt改變，紅綠黃較清楚；藍色偏灰，邊緣模糊並有格紋。這是初步條件效果的定性觀察，尚不能宣稱固定模板已達80%門檻、一般圖示能力或自然語言理解。

## 結論與後續

本輪訓練與Python推論流程已實跑，品質尚未達標。保留候選檢查點及失敗樣本，不提升為預設正式生圖模型。下一步先診斷偏色、邊界與重建／KL的取捨，再提出有明確範圍的新一輪方案；任何再訓練仍須使用者另行確認。網頁與桌面GUI仍待Python品質確認後再延伸。
