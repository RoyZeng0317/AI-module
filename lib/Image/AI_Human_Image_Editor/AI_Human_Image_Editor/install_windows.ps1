$ErrorActionPreference = "Stop"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "找不到 Python Launcher。請先安裝 Python 3.11 或 3.12，並勾選 Add Python to PATH。"
}
py -3.11 -m venv .venv
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "虛擬環境建立失敗。"
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& ".venv\Scripts\python.exe" -m pip install -e ".[dev]"
Write-Host "`n安裝完成。啟用環境：.\\.venv\\Scripts\\Activate.ps1"
Write-Host "查看指令：aihe --help"
