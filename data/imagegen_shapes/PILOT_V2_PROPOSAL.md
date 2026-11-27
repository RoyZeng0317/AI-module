# 第二輪提案：待個別核准，尚未訓練

## 固定條件

沿用原64×64、4色×3形的單物件固定中文模板及資料v1。train240／dev60，test繼續封存；不更改教材、切分、評分器或門檻。

- train manifest SHA256：`1a68c789684c92280f60c0c6baee8025c702b6c76eeb8a8d1b756e5a24680944`。
- dev manifest SHA256：`e0386de274ea83722d6d812597b54b3a22a24625ed0ea765649eaa4fecf9e14e`。
- 固定評分程式SHA256：`778dbb2e6e914f64a37ee5c44ecd2b2ae6709b624af99c2426c39c16418d3783`。
- 首輪model.pt保留：`7ed7696177a9e44dd3089b81f9208d05ef4d37eecea9efdf1c6060b6de7ec0ae`。

## 唯一候選

從零訓練版本化v2 CVAE；原文字編碼與encoder保留候選設計。decoder由轉置卷積改成nearest上採樣＋3×3卷積，輸出仍是模型原始RGB sigmoid，不作palette snapping、銳化、二值化或模板重畫。

loss採前景加權L1＋KL：`d=max_channel(1-target)`，`w=1+4*d`，每張將w除以其全圖平均；`reconstruction=sum(w*abs(output-target))/batch`，`total=reconstruction+1*KL`。target只在訓練loss提供權重，推論時不讀教材。保留與v1相同的pixel sum per batch尺度，分項記錄；L1與BCE數值不可直接當相同品質量尺。

只選這個架構／loss bundle，沒有超參數掃描；即使改善，也不能單靠此輪分離兩項改動的因果。診斷與文獻支持試驗方向，不保證達標。

## 執行上限

RTX4060本機CUDA、FP32、batch16、latent32、cond64、seed42、lr0.001、beta1、foreground_weight4；最多180epochs／2700次更新（每epoch15批），patience30。600秒為批次邊界停止預算，不能中斷單一kernel；OOM／NaN／預算或錯誤就停，不自動改參數重跑。

另存全新 `lib/img_runs/t2i_shapes/pilot_v2`；非空目錄拒絕覆寫。不覆蓋v1，不切換任何正式模型。記錄牆鐘時間、裝置及本程序CUDA峰值配置／保留記憶體（非整張GPU總用量）。

```powershell
.venv/Scripts/python.exe lib/img.py --train-t2i data/imagegen_shapes/v1/train.json --val-manifest data/imagegen_shapes/v1/dev.json --out-dir lib/img_runs/t2i_shapes/pilot_v2 --epochs 180 --batch-size 16 --image-size 64 --latent-dim 32 --cond-dim 64 --lr 0.001 --beta 1 --seed 42 --patience 30 --max-seconds 600 --max-steps 2700 --decoder-kind resize_conv --loss-kind foreground_l1 --foreground-weight 4
```

## 不挑樣本的驗收

1. 已見組合及dev各prompt使用seed42／43／44，逐張原始PNG與逐組分數全部保留。每組至少80%，三個seed下等同3/3；任何組失敗不開test。
2. 全部過關後，dev另用預先鎖定101／102／103確認，不能看結果後換seed。
3. 只有前兩階段都過，才首次test，以seed42／43／44封存驗收。保持各組／各集合分開報告並人工檢查原圖。只適用此窄範圍模板，不宣稱一般自然語言或照片生圖。

本提案準備完且獨立審查後，依待辦#43「每次要跑訓練前都先問」取得確認才執行；未核准前不更新權重。
