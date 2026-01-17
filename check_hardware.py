import os
import sys
import psutil
import multiprocessing
import torch

def validate_setup():
    print("="*50)
    print("🚀 LLIX HARDWARE & ENVIRONMENT VALIDATION")
    print("="*50)

    # 1. Python Version Check (Strikt auf 3.12)
    py_version = sys.version.split()[0]
    is_312 = py_version.startswith("3.12")
    status = "✅ OK" if is_312 else "❌ FALSCHE VERSION (Nutze Python 3.12)"
    print(f"Python Version:  {py_version:<15} {status}")

    # 2. CPU & Threads Check (Optimiert für 16C/32T)
    logical_cores = multiprocessing.cpu_count()
    physical_cores = psutil.cpu_count(logical=False)
    print(f"CPU Kerne:       {physical_cores} Physisch / {logical_cores} Threads")
    
    # 3. RAM Check (Optimiert für 128GB LPDDR5x)
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    print(f"Gesamt-RAM:      {total_gb:.2f} GB")
    if total_gb > 100:
        print(f"RAM-Status:      ✅ High-Memory System erkannt")

    # 4. Torch & Library Check
    print(f"Torch Version:   {torch.__version__}")
    print(f"Intra-op Threads: {torch.get_num_threads()} (Paralleles Rechnen)")
    
    print("="*50)

if __name__ == "__main__":
    validate_setup()
