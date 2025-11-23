# Rollout Profiling 代码重构总结

## 目录结构

```
rollout_profiling/
├── __init__.py
├── workers/
│   ├── __init__.py
│   ├── verl_worker.py      # VERL框架版本（使用ActorRolloutRefWorker）
│   └── vllm_worker.py       # vLLM直接版本（VERL模拟器）
├── utils/
│   ├── __init__.py
│   ├── ema_predictor.py     # EMA长度预测器
│   ├── scheduler.py         # 任务调度器
│   ├── gpu_monitor.py       # GPU监控
│   ├── bin_packing.py       # Bin Packing调度器
│   └── verl_utils.py        # VERL工具函数（配置、DataProto转换等）
└── REFACTORING_SUMMARY.md   # 本文档
```

## 两个版本的区别

### 1. VERL框架版本 (`verl_worker.py`)
- **使用**: `ActorRolloutRefWorker` from VERL
- **数据格式**: `DataProto`
- **配置**: 使用VERL的OmegaConf配置系统
- **优点**: 完全符合VERL规范，可以无缝集成到VERL训练流程
- **适用场景**: 需要与VERL训练流程集成的场景

### 2. vLLM直接版本 (`vllm_worker.py`)
- **使用**: 直接调用 `vLLM.LLM`
- **数据格式**: 直接使用字符串和tokenizer
- **配置**: 直接使用vLLM的参数
- **优点**: 更简单直接，不依赖VERL框架
- **适用场景**: 独立运行profiling，不需要VERL框架的场景

## 公共工具模块

所有两个版本共享的工具类：

1. **`ema_predictor.py`**: EMA长度预测器，用于负载均衡
2. **`scheduler.py`**: 任务调度器（Ray remote），支持实时EMA更新
3. **`gpu_monitor.py`**: GPU监控工具
4. **`bin_packing.py`**: Bin Packing调度器，优化负载均衡

## 使用方式

### 方式1: 使用重构后的模块（推荐）

```python
from rollout_profiling.workers.verl_worker import VERLRolloutWorker
from rollout_profiling.workers.vllm_worker import VLLMRolloutWorker
from rollout_profiling.utils.ema_predictor import EMALengthPredictor
from rollout_profiling.utils.scheduler import TaskScheduler
from rollout_profiling.utils.gpu_monitor import GPUMonitor
from rollout_profiling.utils.bin_packing import BinPackingScheduler
```

### 方式2: 继续使用原有文件

原有的 `enhanced_rollout_profiling.py` 文件仍然保留，可以通过 `--use_verl` 参数选择使用哪个版本。

## 迁移计划

1. ✅ 创建目录结构
2. ✅ 提取公共工具类
3. ✅ 创建VERL版本的worker
4. ⏳ 创建vLLM版本的worker
5. ⏳ 创建统一的入口文件
6. ⏳ 更新原有文件，使其调用新模块

## 下一步

需要完成：
1. 创建 `vllm_worker.py` - vLLM直接版本的worker
2. 创建统一的入口文件，通过参数选择使用哪个版本
3. 更新 `enhanced_rollout_profiling.py`，使其调用新模块（可选）

