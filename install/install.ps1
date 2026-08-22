<#
.SYNOPSIS
    AI-module 一鍵安裝腳本（Windows / PowerShell）。

.DESCRIPTION
    把 README.MD「環境安裝」一節手動打的指令，加上 CLAUDE.md 記錄的既有環境缺口
    （ErrorLog.md 第 8、10、11 條：beautifulsoup4、fitz(PyMuPDF)、prompt_toolkit），
    打包成一支可重複執行的腳本：
      1. 執行 app/install/install_check.py 做硬體前置檢測（可用 -SkipHardwareCheck 跳過）
      2. 建立 / 重用專案根目錄的 .venv
      3. 安裝 torch（預設抓 CUDA 12.8 wheel，對應本專案 RTX 4060/5060 8GB 訓練用顯卡；
         沒有 NVIDIA 顯卡就加 -Cpu 改裝 CPU 版）
      4. 安裝 web/backend/requirements.txt、shape-vision/requirements.txt
      5. 補裝 requirements.txt 沒涵蓋、但 tranning/ 與 lib/components/cli.py 需要的套件
         （pymupdf、prompt_toolkit）
      6. -Dev 額外安裝打包 install_check.exe 用的 pyinstaller/markdown

    只碰這個專案自己的 .venv，不會動 .env、不會呼叫任何外部 AI API（Rule 06）。

.PARAMETER Cpu
    安裝 CPU 版 torch（沒有 NVIDIA 顯卡，或不需要在這台機器上訓練時使用）。
    預設會安裝 CUDA 12.8 版 torch。

.PARAMETER Dev
    額外安裝開發用套件（pyinstaller、markdown），供打包 install_check.exe 等場合使用。

.PARAMETER SkipHardwareCheck
    跳過 install_check.py 硬體前置檢測，直接進入安裝步驟。

.PARAMETER Force
    若 .venv 已存在，先整個刪除再重新建立（乾淨安裝）。預設會重用既有 .venv。

.EXAMPLE
    .\install\install.ps1
    標準安裝：硬體檢測 -> 建立/重用 .venv -> 裝 CUDA 版 torch + 全部相依套件。

.EXAMPLE
    .\install\install.ps1 -Cpu -SkipHardwareCheck
    沒有獨立顯卡的機器，只想跑 sinco 聊天（不訓練）時使用。
#>

[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$Dev,
    [switch]$SkipHardwareCheck,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# install.ps1 放在 install/ 底下，專案根目錄是它的上一層，
# 不管使用者從哪個資料夾呼叫這支腳本都能正確定位。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-PythonCommand {
    foreach ($candidate in @("py", "python")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { return $candidate }
    }
    throw "找不到 Python，請先安裝 Python 3.10 以上版本（https://www.python.org/downloads/）再重新執行本腳本。"
}

Write-Host "========================================================" -ForegroundColor DarkGray
Write-Host " AI-module 安裝腳本" -ForegroundColor DarkGray
Write-Host "========================================================" -ForegroundColor DarkGray
Write-Host "專案根目錄：$ProjectRoot"

$PythonCmd = Get-PythonCommand
$pythonVersion = & $PythonCmd --version
Write-Host "偵測到：$pythonVersion（$PythonCmd）"

# --- 1. 硬體前置檢測 -----------------------------------------------------
if (-not $SkipHardwareCheck) {
    Write-Step "硬體前置檢測（app/install/install_check.py）"
    $installCheckPy = Join-Path $ProjectRoot "app\install\install_check.py"
    if (Test-Path $installCheckPy) {
        & $PythonCmd $installCheckPy
        $proceed = Read-Host "`n是否繼續安裝？(Y/n)"
        if ($proceed -match "^[Nn]") {
            Write-Host "已依你的選擇中止安裝。" -ForegroundColor Yellow
            exit 0
        }
    } else {
        Write-Host "找不到 $installCheckPy，略過硬體檢測。" -ForegroundColor Yellow
    }
} else {
    Write-Step "已指定 -SkipHardwareCheck，跳過硬體前置檢測"
}

# --- 2. 建立 / 重用 .venv -------------------------------------------------
Write-Step "設定虛擬環境（.venv）"
if ((Test-Path $VenvPath) -and $Force) {
    Write-Host "指定 -Force，刪除既有 .venv 後重新建立…" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "建立新的虛擬環境：$VenvPath"
    & $PythonCmd -m venv $VenvPath
} else {
    Write-Host "偵測到既有虛擬環境，直接重用：$VenvPath"
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw ".venv 建立失敗，找不到 $VenvPython"
}

# --- 3. 升級 pip -----------------------------------------------------------
Write-Step "升級 pip"
& $VenvPython -m pip install --upgrade pip

# --- 4. 安裝 torch -----------------------------------------------------------
Write-Step "安裝 torch"
if ($Cpu) {
    Write-Host "指定 -Cpu，安裝 CPU 版 torch。"
    & $VenvPython -m pip install torch
} else {
    Write-Host "安裝 CUDA 12.8 版 torch（本專案訓練算力上限為 RTX 4060/5060 8GB）。"
    Write-Host "沒有 NVIDIA 顯卡的機器請改用 -Cpu 參數重新執行。"
    & $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu128
}

# --- 5. 安裝各元件 requirements.txt ----------------------------------------
Write-Step "安裝 web/backend/requirements.txt（物件偵測 / API / tranning 共用相依）"
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "web\backend\requirements.txt")

Write-Step "安裝 shape-vision/requirements.txt（獨立子專案）"
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "shape-vision\requirements.txt")

# --- 6. 補裝已知環境缺口（見 CLAUDE.md／ErrorLog.md 第 8、10、11 條）---------
Write-Step "補裝已知環境缺口：pymupdf（tranning/ 的 fitz）、prompt_toolkit（lib/components/cli.py）"
& $VenvPython -m pip install pymupdf prompt_toolkit

# --- 7. 開發用套件（可選）---------------------------------------------------
if ($Dev) {
    Write-Step "安裝開發用套件（-Dev）：pyinstaller、markdown"
    & $VenvPython -m pip install pyinstaller markdown
}

# --- 完成 --------------------------------------------------------------------
Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host " 安裝完成" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host @"
接下來可以在專案根目錄執行：

  .venv\Scripts\activate        # 啟用虛擬環境
  python -m lib.main            # 啟動桌面 GUI
  python lib\components\cli.py  # 啟動終端機 CLI
  pytest tranning\               # 跑測試

詳細規則與待辦見 CLAUDE.md、to_do_list.md。
"@
