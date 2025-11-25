#!/usr/bin/env python3
"""
调试辅助工具：在关键位置添加ipdb断点
使用方式：
1. 在代码中导入：from utils.debug_helper import set_debug_breakpoints
2. 调用：set_debug_breakpoints() 来启用断点
3. 或者设置环境变量：export VERL_DEBUG=1
"""

import os
import sys

# 检查是否启用调试模式
DEBUG_ENABLED = os.getenv("VERL_DEBUG", "0") == "1"

def breakpoint_if_debug(label="", **kwargs):
    """
    条件断点：只在DEBUG_ENABLED=True时触发
    
    Args:
        label: 断点标签，用于标识位置
        **kwargs: 要检查的变量
    """
    if DEBUG_ENABLED:
        import ipdb
        print(f"\n{'='*70}")
        print(f"🐛 DEBUG BREAKPOINT: {label}")
        print(f"{'='*70}")
        if kwargs:
            print("Variables:")
            for k, v in kwargs.items():
                print(f"  {k}: {type(v).__name__} = {v}")
        print(f"{'='*70}\n")
        ipdb.set_trace()

def set_debug_breakpoints():
    """启用调试模式"""
    global DEBUG_ENABLED
    DEBUG_ENABLED = True
    os.environ["VERL_DEBUG"] = "1"
    print("✓ Debug mode enabled. Breakpoints will be active.")

def disable_debug_breakpoints():
    """禁用调试模式"""
    global DEBUG_ENABLED
    DEBUG_ENABLED = False
    os.environ["VERL_DEBUG"] = "0"
    print("✓ Debug mode disabled.")

# 自动检查环境变量
if os.getenv("VERL_DEBUG", "0") == "1":
    DEBUG_ENABLED = True



