param([int]$Port = 8060)
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $repoRoot
& "$repoRoot/.venv/Scripts/python.exe" -m uvicorn web.image_processing.app:app --host 127.0.0.1 --port $Port
