"""AI-module 安裝前置檢測。

在使用者安裝/執行本專案（sinco 聊天模型、道路標誌/電路圖辨識訓練等）之前，
先檢查這台電腦的硬體夠不夠。刻意只用標準函式庫（+ 呼叫外部的 nvidia-smi），
不 import torch、不依賴任何 pip 套件 —— 這樣打包出來的 exe 在使用者連
Python 都還沒裝的全新電腦上也能直接跑，這正是「安裝前」檢測該有的樣子。

執行：
    python install_check.py
打包：
    pyinstaller --onefile --console --name install_check install_check.py
"""

import ctypes
import os
import platform
import shutil
import subprocess

MIN_RAM_GB = 8
RECOMMENDED_RAM_GB = 16
MIN_DISK_FREE_GB = 10
RECOMMENDED_DISK_FREE_GB = 20
MIN_CPU_CORES = 4
RECOMMENDED_CPU_CORES = 8
# 呼應專案規則：本專案的訓練算力以 RTX 4060 8GB 為設計上限
RECOMMENDED_VRAM_GB = 8
MIN_USABLE_VRAM_GB = 2


def get_ram_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("sullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys / (1024 ** 3)


def get_disk_free_gb(path: str = "C:\\") -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def get_cpu_cores() -> int:
    return os.cpu_count() or 1


def get_gpu_info() -> list[tuple[str, float]]:
    """回傳 [(顯卡名稱, VRAM GB), ...]；沒有 NVIDIA 顯卡或驅動沒裝時回傳 []。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0:
        return []

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        name, mem_mb = parts
        try:
            gpus.append((name, float(mem_mb) / 1024))
        except ValueError:
            continue
    return gpus


def check_line(label: str, actual: float, minimum: float, recommended: float, unit: str = "") -> bool:
    # 純中文字標記，不用打勾／打叉那類 Unicode 符號字元 —— 那些字在標準
    # Windows 中文版主控台預設的 cp950 編碼裡不存在，直接印出會讓整支 exe
    # 當掉（已實測驗證）。
    if actual >= recommended:
        status = "達建議"
    elif actual >= minimum:
        status = "堪用　"
    else:
        status = "不足　"
    print(f"[{status}] {label}：{actual:.1f}{unit}（最低 {minimum:g}{unit} / 建議 {recommended:g}{unit}）")
    return actual >= minimum


def main() -> None:
    print("=" * 56)
    print("AI-module 安裝前置檢測（sinco 執行/訓練環境）")
    print("=" * 56)
    print(f"作業系統：{platform.system()} {platform.release()} ({platform.machine()})")
    print()

    ok = True
    ok &= check_line("CPU 核心數", get_cpu_cores(), MIN_CPU_CORES, RECOMMENDED_CPU_CORES, " 核")
    ok &= check_line("系統記憶體 (RAM)", get_ram_gb(), MIN_RAM_GB, RECOMMENDED_RAM_GB, " GB")
    ok &= check_line("C 槽可用空間", get_disk_free_gb(), MIN_DISK_FREE_GB, RECOMMENDED_DISK_FREE_GB, " GB")

    print()
    gpus = get_gpu_info()
    gpu_ok = False
    if not gpus:
        print("[未偵測到] 顯示卡：找不到 NVIDIA 顯示卡（或未安裝／未更新顯示卡驅動）")
        print("           -> sinco 聊天功能純 CPU 也能跑，")
        print("              但道路標誌／電路圖辨識這類模型訓練會非常慢，不建議在這台機器上訓練。")
    else:
        for name, vram in gpus:
            # nvidia-smi 回報的 memory.total 會比廠商標示的容量少一點點（保留區塊、
            # MiB/GiB 換算），例如「8GB」卡實際常回報 7.9x GB，四捨五入到整數再比較
            # 門檻，才不會讓專案自己指名的 RTX 4060 8GB 被誤判成「僅堪用」。
            if round(vram) >= RECOMMENDED_VRAM_GB:
                status = "達建議"
            elif vram >= MIN_USABLE_VRAM_GB:
                status = "堪用　"
            else:
                status = "不足　"
            print(f"[{status}] 顯示卡：{name}（{vram:.1f} GB VRAM，本專案訓練以 RTX 4060 8GB 為設計上限）")
            gpu_ok = gpu_ok or vram >= MIN_USABLE_VRAM_GB

    print()
    print("=" * 56)
    if ok and gpu_ok:
        print("結論：適合安裝本專案，硬體足以執行 sinco 並進行模型訓練。")
    elif ok:
        print("結論：適合安裝執行 sinco 聊天等輕量功能；")
        print("      但沒有可用的獨立顯示卡，不建議在這台機器上做模型訓練。")
    else:
        print("結論：不適合安裝本專案 —— 未達建議的最低系統需求，執行可能會很吃力或不穩定。")
    print("=" * 56)
    input("\n按 Enter 鍵結束...")


if __name__ == "__main__":
    main()
