#!/usr/bin/env python3
"""
在关键位置添加ipdb断点的脚本
使用方法：
1. 设置环境变量：export VERL_DEBUG=1
2. 运行实验，会在断点处暂停
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def add_breakpoint_to_file(file_path, pattern, insert_after=True, label=""):
    """
    在文件中添加断点代码
    
    Args:
        file_path: 文件路径
        pattern: 要匹配的模式（正则表达式）
        insert_after: True表示在匹配行之后插入，False表示在匹配行之前插入
        label: 断点标签
    """
    file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        print(f"⚠️  文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经添加过断点
    if f'# DEBUG_BREAKPOINT_{label}' in content:
        print(f"✓ {file_path.name} 中已存在断点: {label}")
        return True
    
    # 查找匹配的行
    lines = content.split('\n')
    new_lines = []
    inserted = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # 检查是否匹配模式
        if re.search(pattern, line) and not inserted:
            # 添加断点代码
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            if insert_after:
                # 在匹配行之后插入
                debug_code = f'''{indent_str}# DEBUG_BREAKPOINT_{label}
{indent_str}import os
{indent_str}if os.getenv("VERL_DEBUG", "0") == "1":
{indent_str}    import ipdb
{indent_str}    print("\\n{'='*70}")
{indent_str}    print(f"🐛 DEBUG BREAKPOINT: {label}")
{indent_str}    print(f"{'='*70}\\n")
{indent_str}    ipdb.set_trace()'''
                new_lines.append(debug_code)
            else:
                # 在匹配行之前插入（需要先移除当前行，插入代码，再添加当前行）
                new_lines.pop()  # 移除刚添加的当前行
                debug_code = f'''{indent_str}# DEBUG_BREAKPOINT_{label}
{indent_str}import os
{indent_str}if os.getenv("VERL_DEBUG", "0") == "1":
{indent_str}    import ipdb
{indent_str}    print("\\n{'='*70}")
{indent_str}    print(f"🐛 DEBUG BREAKPOINT: {label}")
{indent_str}    print(f"{'='*70}\\n")
{indent_str}    ipdb.set_trace()'''
                new_lines.append(debug_code)
                new_lines.append(line)  # 重新添加当前行
            
            inserted = True
    
    if inserted:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print(f"✓ 已在 {file_path.name} 添加断点: {label}")
        return True
    else:
        print(f"⚠️  在 {file_path.name} 中未找到匹配模式: {pattern}")
        return False

def main():
    """在关键位置添加断点"""
    print("=" * 70)
    print("添加调试断点")
    print("=" * 70)
    
    breakpoints = [
        # TaskRunner.run 方法开始
        {
            "file": "verl/verl/trainer/main_ppo.py",
            "pattern": r'def run\(self, config\):',
            "label": "TaskRunner.run",
            "insert_after": True
        },
        # RayPPOTrainer.fit 方法开始
        {
            "file": "verl/verl/trainer/ppo/ray_trainer.py",
            "pattern": r'def fit\(self\):',
            "label": "RayPPOTrainer.fit",
            "insert_after": True
        },
        # RayPPOTrainer._validate 方法开始
        {
            "file": "verl/verl/trainer/ppo/ray_trainer.py",
            "pattern": r'def _validate\(self\):',
            "label": "RayPPOTrainer._validate",
            "insert_after": True
        },
        # RayPPOTrainer._apply_scheduling 方法开始
        {
            "file": "verl/verl/trainer/ppo/ray_trainer.py",
            "pattern": r'def _apply_scheduling\(self, gen_batch: DataProto\) -> DataProto:',
            "label": "RayPPOTrainer._apply_scheduling",
            "insert_after": True
        },
        # _validate 中应用调度后
        {
            "file": "verl/verl/trainer/ppo/ray_trainer.py",
            "pattern": r'test_batch = self\._apply_scheduling_validation\(test_batch\)',
            "label": "_validate_after_scheduling",
            "insert_after": True
        },
        # fit 中应用调度后
        {
            "file": "verl/verl/trainer/ppo/ray_trainer.py",
            "pattern": r'gen_batch = self\._apply_scheduling\(gen_batch\)',
            "label": "fit_after_scheduling",
            "insert_after": True
        },
    ]
    
    success_count = 0
    for bp in breakpoints:
        if add_breakpoint_to_file(
            bp["file"],
            bp["pattern"],
            bp["insert_after"],
            bp["label"]
        ):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"完成: {success_count}/{len(breakpoints)} 个断点已添加")
    print("=" * 70)
    print("\n使用方法:")
    print("  1. 设置环境变量: export VERL_DEBUG=1")
    print("  2. 运行实验脚本")
    print("  3. 程序会在断点处暂停，进入ipdb调试器")
    print("  4. 使用 ipdb 命令进行调试:")
    print("     - n (next): 执行下一行")
    print("     - s (step): 进入函数")
    print("     - c (continue): 继续执行")
    print("     - p <变量名>: 打印变量")
    print("     - pp <变量名>: 美化打印变量")
    print("     - l (list): 显示当前代码")
    print("     - u (up): 向上移动栈帧")
    print("     - d (down): 向下移动栈帧")
    print("     - q (quit): 退出调试器")

if __name__ == "__main__":
    main()



