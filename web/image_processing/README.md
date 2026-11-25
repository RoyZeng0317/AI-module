# 本機影像處理實驗室

在專案根目錄執行 `./web/image_processing/run_local.ps1`，開啟 http://127.0.0.1:8060 。亦可執行：

```powershell
.venv/Scripts/python.exe -m uvicorn web.image_processing.app:app --host 127.0.0.1 --port 8060
```

依賴見 requirements.txt；不需要 API key。這是本機網頁，並非 Codex Cloud 託管；重開機後須重新啟動。數值處理重用 `lib/opencv2/` 的純 Python for 迴圈，沒有 NumPy、OpenCV 或雲端模型。

選擇圖片、卷積或直方圖均衡化，再按「開始處理」。可下載 PNG、原始數值 JSON 或直方圖 CSV。卷積核翻轉 180 度，Filter 1 不除以 8；PNG 的裁切／min–max 僅供顯示，JSON 保留帶符號數值。均衡化比較的是亮度 Y 與均衡化 Y，並提供 Cb／Cr、CDF 與映射表，不是彩色重建。

限制為 10 MiB、1,048,576 像素；支援 PNG/JPEG/WebP/TIFF/BMP，取第一影格。透明以白底合成，保留解碼像素方向，不自動套用 EXIF 旋轉或縮圖。edge／zero 保持尺寸，valid 縮小。純迴圈可能需要等待；一次處理一項，最多兩項排隊。取消瀏覽器等待不會取消已提交的伺服器工作。

結果保存在專案 `output/image-processing-web/jobs/`，不會自動刪除。停止服務後可自行清理該結果目錄；重啟後舊工作網址失效，檔案仍在磁碟。伺服器只綁定本機回環位址，不設定開機啟動。測試與審查紀錄見 `team-run/image-web/`。
