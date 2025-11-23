#!/usr/bin/env python3
"""
将bin packing调度功能应用到实际的verl安装目录
"""

import shutil
import os

# 源文件路径（rollout_profiling_system中的文件）
source_files = {
    'bin_packing_scheduler.py': '/data/250010176/codes/rollout_profiling_system/verl/verl/utils/bin_packing_scheduler.py',
    'bin_packing_dispatch.py': '/data/250010176/codes/rollout_profiling_system/verl/verl/single_controller/base/bin_packing_dispatch.py',
}

# 目标文件路径（实际的verl安装目录）
target_files = {
    'bin_packing_scheduler.py': '/data/250010176/codes/verl/verl/utils/bin_packing_scheduler.py',
    'bin_packing_dispatch.py': '/data/250010176/codes/verl/verl/single_controller/base/bin_packing_dispatch.py',
}

# 复制文件
print("Copying bin packing files to verl installation...")
for name, source in source_files.items():
    target = target_files[name]
    target_dir = os.path.dirname(target)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(source, target)
    print(f"  ✓ Copied {name} to {target}")

print("\n✅ All files copied successfully!")
print("\nNote: You still need to manually patch ray_trainer.py with the bin packing methods.")
print("The methods are:")
print("  - _apply_bin_packing_schedule()")
print("  - _apply_bin_packing_schedule_validation()")
print("And add calls to these methods in:")
print("  - fit() method (after gen_batch_output.repeat())")
print("  - _validate() method (after test_batch.repeat())")


