# 2026-08-27

## WinError 10048：port 8000 already in use
啟動 `web/backend/app.py` 時綁定 0.0.0.0:8000 失敗。
`netstat -ano` 查到 PID 20976 已經在監聽 8000，`Get-CimInstance Win32_Process`
確認該行程是先前啟動的
`C:\Users\Roy\.anaconda-desktop\micromamba\envs\cuda\python.exe app.py`
（用的是 ErrorLog.md 第 11 條提到的 `cuda` conda 環境）。
不是程式碼問題，是同一支 app.py 被重複啟動、舊行程沒關掉。

## 瀏覽器主控台 pyodide 錯誤（GPU info check failed: get / borrowed proxy destroyed）
`web/frontend/src/components/action.py` 目前原始碼裡這兩個問題都已經有對應修正
（`api_get()` 有 `.to_py()`；`capture_and_send()` 的 `toBlob` resolve 有包
`create_once_callable()`，程式碼註解本身就是在講這兩個修正的來龍去脈）。
主控台仍看到這兩個錯誤訊息，研判是瀏覽器分頁快取了修正前的舊版 action.py
（FastAPI 端 `public_file()` 每次請求都用 `FileResponse` 直接讀硬碟現在的檔案，
後端本身沒有快取問題）。尚待使用者確認：硬重新整理（Ctrl+Shift+R）或開
DevTools 停用快取後重新載入頁面，是否還會出現同樣錯誤。
