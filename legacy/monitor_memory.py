#!/usr/bin/env python3
"""
Memory monitoring script for VSlite training
Run this in a separate terminal to monitor GPU memory usage
"""

import time
import subprocess
import re

def get_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                               '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for i, line in enumerate(lines):
            used, total = map(int, line.split(', '))
            gpu_info.append((i, used, total))
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return []

def main():
    print("GPU Memory Monitor - Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while True:
            gpu_info = get_gpu_memory()
            if gpu_info:
                for gpu_id, used, total in gpu_info:
                    usage_percent = (used / total) * 100
                    print(f"GPU {gpu_id}: {used:5d}MB / {total:5d}MB ({usage_percent:5.1f}%)")
                print("-" * 50)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
