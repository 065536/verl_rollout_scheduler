#!/usr/bin/env python3
"""
GPU监控
"""

import time
import threading
import subprocess


class GPUMonitor:
    """GPU监控"""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        self.gpu_data = []
        self.lock = threading.Lock()
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("✓ GPU monitoring started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("✓ GPU monitoring stopped")
        return self.gpu_data
    
    def _monitor_loop(self):
        while self.running:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    timestamp = time.time()
                    gpu_info = {'timestamp': timestamp, 'gpus': []}
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 5:
                                gpu_info['gpus'].append({
                                    'index': int(parts[0]),
                                    'util_gpu': float(parts[1]),
                                    'memory_used': float(parts[3]),
                                })
                    with self.lock:
                        self.gpu_data.append(gpu_info)
            except:
                pass
            time.sleep(self.interval)

