# 把桌面 GUI 打包成 app/install/install.msi（WiX 5，個人安裝到 %LOCALAPPDATA%\AI-module）。
# 只複製 GUI 執行需要的程式碼與正式使用中的模型權重；不含 .venv / .env（Rule 05）/ 測試 / 訓練資料集。
# .venv 由安裝完自動執行的 install/install.ps1 在目標電腦上建立（需要網路）。
# 用法：.\install\build_msi.ps1 [-Version 1.0.0]
param([string]$Version = "1.0.0")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Stage = Join-Path $Root "build\msi\stage"
$Msi = Join-Path $Root "app\install\install.msi"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }

# 整個資料夾複製（排除 .env、測試、快取）
$xf = ".env", "*.env", "test_*.py", ".cli_history", "*.pyc"; $xd = "__pycache__", ".pytest_cache"
$dirs = "lib", "app\command", "tranning\MentalHealth", "tranning\characters", "tranning\chat_runs", "tranning\code_runs",
        "tranning\gpt_chat_runs", "tranning\image_classifier_runs", "tranning\speech_runs", "tranning\yolo_runs\custom\weights"
foreach ($d in $dirs) {
    robocopy (Join-Path $Root $d) (Join-Path $Stage $d) /E /XF $xf /XD $xd /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy 失敗：$d" }
}

# 單檔複製（保留相對路徑）
$files = "tranning\*.py", "web\backend\*.py", "web\backend\requirements.txt", "shape-vision\requirements.txt",
         "app\install\install_check.exe", "app\install\install_check.py", "install\install.ps1",
         "data\pairs.json", "data\code_pairs.json", "data\mental.csv", "data\character*.json", "data\auto_learn_candidates.json",
         "yolo11n.pt", "yolo26n.pt", "*.task", "model_icon.ico"
foreach ($f in $files) {
    $dst = "$Stage\$(Split-Path $f -Parent)"; New-Item -ItemType Directory -Force $dst | Out-Null
    Get-ChildItem (Join-Path $Root $f) | Where-Object Name -notlike "test_*" | Copy-Item -Destination $dst
}

$size = (Get-ChildItem $Stage -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("暫存區完成：{0:N1} MB -> 開始編譯 MSI" -f $size) -ForegroundColor Cyan
wix build (Join-Path $PSScriptRoot "AI-module.wxs") -arch x64 -d Version=$Version -bindpath stage=$Stage -o $Msi
if ($LASTEXITCODE -ne 0) { throw "wix build 失敗（exit $LASTEXITCODE）" }
Remove-Item ([IO.Path]::ChangeExtension($Msi, ".wixpdb")) -ErrorAction SilentlyContinue
Write-Host ("完成：{0}（{1:N1} MB）" -f $Msi, ((Get-Item $Msi).Length / 1MB)) -ForegroundColor Green
