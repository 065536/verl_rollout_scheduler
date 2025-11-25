# 调试指南 - 使用 ipdb 进行断点调试

## 概述

已在本项目的关键位置添加了 ipdb 断点，可以通过环境变量 `VERL_DEBUG=1` 来启用调试模式。

## 已添加的断点位置

1. **TaskRunner.run** (`verl/verl/trainer/main_ppo.py`)
   - 训练流程的入口点
   - 可以查看配置、worker初始化等

2. **RayPPOTrainer.fit** (`verl/verl/trainer/ppo/ray_trainer.py`)
   - 训练循环的入口点
   - 可以查看训练循环的开始

3. **RayPPOTrainer._validate** (`verl/verl/trainer/ppo/ray_trainer.py`)
   - 验证阶段的入口点
   - 可以查看验证数据的处理

4. **RayPPOTrainer._apply_scheduling** (`verl/verl/trainer/ppo/ray_trainer.py`)
   - 调度策略应用点
   - 可以查看调度前后的数据变化

5. **fit_after_scheduling** (`verl/verl/trainer/ppo/ray_trainer.py`)
   - 训练循环中应用调度后
   - 可以查看调度后的数据状态

## 使用方法

### 方法1: 通过环境变量启用

```bash
# 在运行脚本前设置
export VERL_DEBUG=1
bash scripts/run_batch_scheduling_experiments_single_gpu.sh
```

### 方法2: 在脚本中启用

编辑 `scripts/run_batch_scheduling_experiments_single_gpu.sh`，找到以下行：

```bash
# export VERL_DEBUG=1  # 取消注释以启用调试模式
```

改为：

```bash
export VERL_DEBUG=1  # 启用调试模式
```

### 方法3: 临时启用（仅一次运行）

```bash
VERL_DEBUG=1 bash scripts/run_batch_scheduling_experiments_single_gpu.sh
```

## ipdb 常用命令

当程序在断点处暂停时，可以使用以下命令：

### 基本导航
- `n` (next): 执行下一行代码（不进入函数）
- `s` (step): 进入函数内部
- `c` (continue): 继续执行直到下一个断点
- `l` (list): 显示当前代码上下文
- `ll` (longlist): 显示整个函数代码

### 查看变量
- `p <变量名>`: 打印变量值
- `pp <变量名>`: 美化打印变量（适合复杂对象）
- `vars()`: 查看当前作用域的所有变量
- `locals()`: 查看局部变量
- `globals()`: 查看全局变量

### 栈帧操作
- `u` (up): 向上移动一个栈帧
- `d` (down): 向下移动一个栈帧
- `w` (where): 显示当前栈帧信息
- `bt` (backtrace): 显示完整的调用栈

### 执行代码
- `!<Python代码>`: 执行Python代码
- `interact`: 进入交互式Python shell

### 其他
- `h` (help): 显示帮助信息
- `q` (quit): 退出调试器（会终止程序）
- `restart`: 重新启动调试会话

## 调试示例

### 示例1: 查看调度前后的数据

当程序在 `_apply_scheduling` 断点处暂停时：

```python
# 查看调度前的数据
pp gen_batch.batch  # 查看batch数据
pp len(gen_batch)   # 查看数据长度

# 执行调度
scheduled = self._apply_scheduling(gen_batch)

# 查看调度后的数据
pp scheduled.batch
pp len(scheduled)

# 继续执行
c
```

### 示例2: 查看验证数据

当程序在 `_validate` 断点处暂停时：

```python
# 查看验证数据
pp test_batch
pp len(test_batch)

# 查看配置
pp self.config.trainer.val_before_train

# 继续执行
c
```

### 示例3: 检查调度模式

```python
# 在 _apply_scheduling 断点处
import os
pp os.getenv("VERL_SCHEDULE_MODE")
pp os.getenv("VERL_ENABLE_SHUFFLE")
pp os.getenv("VERL_BIN_PACKING_SCHEDULE_FILE")
```

## 注意事项

1. **调试模式会显著减慢执行速度**：每次遇到断点都会暂停，建议只在需要调试时启用

2. **Ray远程执行**：由于Ray使用远程执行，断点可能出现在不同的进程中。如果断点没有触发，可能是：
   - 代码在Ray worker中执行，而不是主进程
   - 需要检查Ray worker的日志

3. **多进程调试**：在多GPU/多节点环境中，断点可能出现在不同的进程中，需要分别调试

4. **退出调试器**：使用 `q` 退出调试器会终止整个程序，使用 `c` 继续执行

## 移除断点

如果需要移除所有断点，可以运行：

```bash
# 查找并移除所有断点标记
grep -r "DEBUG_BREAKPOINT" verl/verl/trainer/
```

然后手动删除相关代码块，或者使用git恢复：

```bash
git checkout verl/verl/trainer/main_ppo.py
git checkout verl/verl/trainer/ppo/ray_trainer.py
```

## 重新添加断点

如果需要重新添加断点：

```bash
python3 utils/add_debug_breakpoints.py
```

## 故障排除

### 断点没有触发

1. 确认 `VERL_DEBUG=1` 已设置
2. 检查代码是否真的执行到了断点位置
3. 查看日志确认程序运行状态

### ipdb 命令不响应

1. 确认终端支持交互式输入
2. 尝试使用 `Ctrl+C` 然后输入命令
3. 检查是否在Ray worker中（可能需要查看worker日志）

### 无法查看变量

1. 确认变量在当前作用域中
2. 使用 `vars()` 或 `locals()` 查看所有可用变量
3. 使用 `pp` 而不是 `p` 查看复杂对象



