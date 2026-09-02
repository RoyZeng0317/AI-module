"""顯卡（GPU）基本資訊查詢工具。

只給後端（web/backend/app.py）匯入使用——瀏覽器端的 PyScript (Pyodide) 是
WASM 沙盒，摸不到主機的顯卡，nvidia-smi、torch.cuda 這些都無法在瀏覽器裡
執行，所以查詢邏輯得留在後端，查到的結果再由 /api/gpu_info 端點交給
action.py 顯示。這支檔案本身沒有被加進 app.py 的 PUBLIC_FILES，不會被當成
靜態檔案直接送到瀏覽器。
"""

import subprocess

import torch


def _nvidia_smi_query(fields):
    """呼叫 nvidia-smi 查詢指定欄位，任何失敗都回傳 None。

    沒有裝 nvidia-smi、沒有 NVIDIA 顯卡、驅動沒裝好時都會走到這裡，不能讓
    整個 get_gpu_info() 因為這個輔助查詢而炸掉——driver_version/溫度只是
    錦上添花的資訊，缺了就顯示 None，不影響 torch.cuda 那份主要資訊。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        first_line = out.stdout.strip().split("\n")[0]
        return [part.strip() for part in first_line.split(",")]
    except Exception:
        return None


def get_gpu_info():
    """回傳一份顯卡基本資訊的 dict，沒有可用的 CUDA 顯卡時 available=False。"""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "cuda_version": None,
            "driver_version": None,
            "total_memory_mb": None,
            "used_memory_mb": None,
            "temperature_c": None,
        }

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)

    driver_version = None
    temperature_c = None
    smi_fields = _nvidia_smi_query("driver_version,temperature.gpu")
    if smi_fields and len(smi_fields) == 2:
        driver_version = smi_fields[0]
        try:
            temperature_c = float(smi_fields[1])
        except ValueError:
            temperature_c = None

    return {
        "available": True,
        "name": props.name,
        "cuda_version": torch.version.cuda,
        "driver_version": driver_version,
        "total_memory_mb": int(props.total_memory // (1024 * 1024)),
        "used_memory_mb": int(torch.cuda.memory_allocated(idx) // (1024 * 1024)),
        "temperature_c": temperature_c,
    }
