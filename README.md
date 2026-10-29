<h1 align="center">O²-RTDETR × L315_vehicle</h1>
<p align="center"><strong>单卡 Windows 环境下的旋转目标检测基线训练流程</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.0.1">
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white" alt="CUDA 11.8">
  <img src="https://img.shields.io/badge/MMDetection-3.0.0-5C6BC0" alt="MMDetection 3.0.0">
  <img src="https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows&logoColor=white" alt="Windows">
</p>

本文档记录从拉取仓库、创建 Python 环境、准备 `L315_vehicle` 数据集、完成硬件适配，到启动正式基线训练的必要步骤。

> [!NOTE]
> 原项目的论文、模型介绍与官方说明保存在 [README_OFFICIAL.md](README_OFFICIAL.md)。本仓库已经集成运行所需的源码兼容修复。

## 🧭 快速导航

- [🎯 最终基线设置](#1-最终基线设置)
- [📁 目录结构](#2-目录结构)
- [🐍 创建 Conda 环境](#3-创建-conda-环境)
- [📦 安装依赖](#4-安装依赖)
- [🧩 已集成的兼容修复](#5-仓库已集成的兼容修复)
- [⚙️ L315 配置](#6-使用仓库内置的-l315_vehicle-配置)
- [✅ 检查最终配置](#7-检查最终配置)
- [🧪 预训练权重与硬件适配](#8-首次启动预训练权重与硬件适配)
- [🔥 完整数据集压力测试](#9-完整数据集-1-轮压力测试)
- [🚀 正式 72 轮训练](#10-继续正式-72-轮基线训练)
- [⏯️ 中断与恢复](#11-中断与恢复)
- [📊 训练产物与结果记录](#12-训练产物和基线记录)

```mermaid
flowchart LR
    A[拉取仓库] --> B[创建环境]
    B --> C[准备数据集]
    C --> D[下载预训练权重]
    D --> E[测试 Batch 与 Workers]
    E --> F[完整数据跑 1 轮]
    F --> G[正式训练 72 轮]
    G --> H[分析收敛与创新空间]
```

## 1. 🎯 最终基线设置

| 项目 | 设置 |
|---|---|
| 模型 | O²-RTDETR-R18 |
| 数据集 | L315_vehicle |
| 类别 | Bus、Car、Truck、Van |
| 输入画布 | 1024 × 1024 |
| GPU | NVIDIA GeForce RTX 4070 Laptop 8 GB |
| 训练 batch size | 4 |
| 验证/test batch size | 4 |
| 梯度累积 | 2 |
| 有效 batch size | 4 × 2 = 8 |
| num_workers | 2 |
| 数值精度 | FP32，不使用 AMP |
| 优化器 | AdamW |
| 基础学习率 | 1e-4 |
| 训练上限 | 72 epochs |
| 验证间隔 | 6 epochs |
| 权重保存 | 每轮保存 |

## 2. 📁 目录结构

确保主要目录如下：

```text
D:\Object_Detection\
├── O2-RT-DETR-main\
│   ├── tools\
│   ├── ai4rs\
│   └── projects\rotated_rtdetr\
└── data\L315_vehicle\
    ├── train\
    │   ├── images\
    │   └── labelTxt\
    ├── val\
    │   ├── images\
    │   └── labelTxt\
    └── test\
        └── images\
```

当前划分：

```text
train: 12,248 张
val:    3,065 张
test:   3,350 张
```

实际标注中只有四类：`Bus`、`Car`、`Truck`、`Van`。

## 3. 🐍 创建 Conda 环境

在 PowerShell 中执行：

```powershell
conda create -n pytorch201 python=3.9 -y
conda activate pytorch201
cd D:\Object_Detection\O2-RT-DETR-main
```

后续所有命令都在 `pytorch201` 环境和项目根目录下执行。

## 4. 📦 安装依赖

### 4.1 安装 PyTorch 2.0.1 + CUDA 11.8

```powershell
python -m pip install torch==2.0.1 torchvision==0.15.2 `
  --index-url https://download.pytorch.org/whl/cu118
```

### 4.2 安装固定版本的基础依赖

```powershell
python -m pip install `
  numpy==1.26.4 `
  opencv-python==4.10.0.84 `
  mmengine==0.10.7 `
  mmdet==3.0.0 `
  scipy==1.13.1 `
  shapely==2.0.7 `
  pycocotools==2.0.11 `
  terminaltables matplotlib six cython `
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.3 安装 MMCV 2.0.1

```powershell
python -m pip install mmcv==2.0.1 `
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### 4.4 以可编辑方式安装当前项目

```powershell
python -m pip install -e . --no-deps
```

### 4.5 验证环境

```powershell
python -c "import torch,torchvision,numpy,cv2,mmcv,mmengine,mmdet,ai4rs; print('torch:',torch.__version__); print('torchvision:',torchvision.__version__); print('numpy:',numpy.__version__); print('opencv:',cv2.__version__); print('mmcv:',mmcv.__version__); print('mmengine:',mmengine.__version__); print('mmdet:',mmdet.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0))"

python -m pip check
```

应对应：

```text
torch:       2.0.1+cu118
torchvision: 0.15.2+cu118
numpy:       1.26.4
opencv:      4.10.0
mmcv:        2.0.1
mmengine:    0.10.7
mmdet:       3.0.0
CUDA:        True
```

`pip check` 应输出 `No broken requirements found.`

## 5. 🧩 仓库已集成的兼容修复

> [!IMPORTANT]
> 本仓库已经包含运行 O²-RTDETR 所需的兼容性修复。拉取代码后**不需要再次修改这些源码**。

已集成内容包括：

- Python 3.9 类型注解兼容；
- `ConcatDataset` 的 MMEngine 导入适配；
- 可选 SARDet 指标的隔离；
- L315 标注头 `imagesource`、`gsd` 的过滤；
- O²-RTDETR Decoder 与 MMDetection 3.0.0 的接口适配；
- Windows 使用 `spawn` 启动数据加载进程；
- L315 四分类数据加载和训练配置。

这些修改属于运行兼容性适配，不改变 O²-RTDETR 的网络结构和损失定义。

## 6. ⚙️ 使用仓库内置的 L315_vehicle 配置

训练配置已经包含在仓库中：

```text
projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py
```

默认数据路径为：

```text
D:/Object_Detection/data/L315_vehicle/
```

如果的数据集放在其他位置，只需要修改配置中的这一行：

```python
dataset_root = '实际的/L315_vehicle/路径/'
```

仓库中的默认值是 RTX 4070 Laptop 8 GB 上已经验证通过的参考配置。模型、数据划分、输入尺寸和优化器应保持不变；batch size、梯度累积和 workers 必须先按自己的硬件完成第 8 节测试后再确定：

```text
类别：Bus、Car、Truck、Van
输入：1024 × 1024
训练 batch：4
验证/test batch：4
num_workers：2
梯度累积：2
有效 batch：8
精度：FP32
总训练上限：72 epochs
```

不要使用 `--amp`。当前 PyTorch 版本下，项目的 KLD 匹配代价不支持 FP16 矩阵行列式运算。
## 7. ✅ 检查最终配置

```powershell
python -c "from mmengine.config import Config; c=Config.fromfile(r'projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_1xb1_72e_l315.py'); print('train batch:',c.train_dataloader.batch_size); print('val batch:',c.val_dataloader.batch_size); print('workers:',c.train_dataloader.num_workers); print('accumulation:',c.optim_wrapper.accumulative_counts); print('effective batch:',c.train_dataloader.batch_size*c.optim_wrapper.accumulative_counts); print('epochs:',c.train_cfg.max_epochs)"
```

正确输出：

```text
train batch: 4
val batch: 4
workers: 2
accumulation: 2
effective batch: 8
epochs: 72
```

## 8. 🧪 首次启动、预训练权重与硬件适配

> [!WARNING]
> 不同电脑不能直接假定 `batch=4、workers=2` 一定可用。必须先完成小规模测试，再固定该电脑的正式配置。以下测试只覆盖运行参数，不改变模型、输入尺寸、数据集和优化器。

可以把整个过程理解为：

| 阶段 | 类比 | 目的 | 是否评价最终精度 |
|---|---|---|---|
| 32/128 张小规模测试 | 设备检查 | 确认环境、显存、batch 和 workers 可用 | 否 |
| 完整数据 1 轮 | 模考 | 检查全数据、密集目标、长时稳定性和验证流程 | 只看趋势 |
| 固定配置训练 72 轮 | 正式考试 | 建立可用于论文对比的基线 | 是 |

### 8.1 首次启动与预训练权重

R18 配置已经指定官方预训练权重。第一次执行训练命令时会自动下载：

```text
https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rtdetr/resnet18vd_pretrained_55f5a0d6.pth
```

正常日志包含：

```text
load model from: ...resnet18vd_pretrained_55f5a0d6.pth
```

下载后通常缓存到：

```text
C:\Users\<用户名>\.cache\torch\hub\checkpoints\
```

### 8.2 从保守配置开始测试训练

这一步是“设备检查”，不是精度实验。它需要同时验证：

- 数据能否正常读取；
- 模型能否完成前向传播；
- loss 能否正常计算；
- 反向传播和梯度累积是否正常；
- checkpoint 能否保存；
- 验证流程能否完整结束。

先测试物理 batch 1，并保持有效 batch 为 8：

```powershell
python tools\train.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  --cfg-options `
  train_dataloader.batch_size=1 `
  train_dataloader.num_workers=0 `
  train_dataloader.persistent_workers=False `
  train_dataloader.dataset.indices=32 `
  val_dataloader.batch_size=1 `
  val_dataloader.num_workers=0 `
  val_dataloader.persistent_workers=False `
  optim_wrapper.accumulative_counts=8 `
  train_cfg.max_epochs=1 `
  work_dir=./work_dirs/hardware_test_b1
```

32 张图片的预期训练迭代数：

| 训练 batch | 迭代数 | 梯度累积 | 参数更新次数 |
|---:|---:|---:|---:|
| 1 | 32 | 8 | 4 |
| 2 | 16 | 4 | 4 |
| 4 | 8 | 2 | 4 |

终端应出现类似：

```text
Epoch(train) [1][32/32]   # batch 1
Epoch(train) [1][16/16]   # batch 2
Epoch(train) [1][8/8]     # batch 4
Saving checkpoint at 1 epochs
```

> [!CAUTION]
> 32 张图片只能判断“能不能跑”，不能判断模型精度。此时 mAP 为 0 或非常低都是正常现象。

成功后逐级测试，而不是直接使用最大值：

| 物理 batch | 梯度累积 | 有效 batch |
|---:|---:|---:|
| 1 | 8 | 8 |
| 2 | 4 | 8 |
| 4 | 2 | 8 |

测试 batch 2 时，把命令中的对应值改为：

```text
train_dataloader.batch_size=2
optim_wrapper.accumulative_counts=4
work_dir=./work_dirs/hardware_test_b2
```

测试 batch 4 时改为：

```text
train_dataloader.batch_size=4
optim_wrapper.accumulative_counts=2
work_dir=./work_dirs/hardware_test_b4
```

出现 `CUDA out of memory` 时停止提高 batch，使用上一个稳定值。不要通过降低 1024×1024 输入尺寸或开启 AMP 来绕过显存测试；这会改变基线条件，且当前 KLD 匹配计算不支持完整 FP16 训练。

每次 batch 测试记录下列数据：

| batch | accumulation | time/批 | time/张 | memory | 是否 OOM |
|---:|---:|---:|---:|---:|---|
| 1 | 8 |  |  |  |  |
| 2 | 4 |  |  |  |  |
| 4 | 2 |  |  |  |  |

在不 OOM 的候选项中，优先选择物理 batch 较大且每张耗时较低的配置，但至少保留一定显存余量。

### 8.3 测试验证/test batch

验证不保存梯度，通常能使用与训练相同或更大的 batch。使用已经生成的测试权重运行：

```powershell
python tools\test.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  work_dirs\hardware_test_b1\epoch_1.pth `
  --cfg-options `
  test_dataloader.batch_size=4 `
  test_dataloader.num_workers=0 `
  test_dataloader.persistent_workers=False `
  --work-dir work_dirs/val_batch_test
```

如果验证 batch 4 OOM，则依次退回 2、1。验证 batch 只影响吞吐量和显存，不更新模型权重。

使用同一 checkpoint 对比不同验证 batch，并记录：

| 验证 batch | data_time | time | mAP | 是否 OOM |
|---:|---:|---:|---:|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 4 |  |  |  |  |

同一权重下 mAP 应保持一致或只有正常浮点波动。

### 8.4 测试 num_workers

`num_workers` 取决于 CPU、系统内存、硬盘和操作系统。Windows 建议从 0 开始，再测试 2；只有 2 明显更快且稳定时才保留。

```text
num_workers=0  -> persistent_workers=False
num_workers>0  -> persistent_workers=True
```

比较日志中的 `data_time` 和 `time`，选择稳定且总 `time` 最小的值。workers 过多可能因进程调度和磁盘竞争而变慢。

为了避免 worker 启动开销干扰结果，worker 测试建议至少使用 128 张训练图片。比较表：

| workers | persistent_workers | data_time | time | 是否稳定 |
|---:|---|---:|---:|---|
| 0 | False |  |  |  |
| 2 | True |  |  |  |
| 3 | True |  |  |  |
| 4 | True |  |  |  |

不是 workers 越多越快；选择总 `time` 最低且长时运行不报错的值。

### 8.5 固定该电脑的运行配置

完成测试后，把选定值写回 L315 配置文件，并再次执行第 7 节检查。此后正式基线训练期间不再更改：

```text
训练 batch size
梯度累积次数
验证/test batch size
num_workers
persistent_workers
```

必须始终保证：

```text
训练物理 batch × 梯度累积 = 有效 batch 8
```

## 9. 🔥 完整数据集 1 轮压力测试

这一步相当于正式训练前的“模考”。小规模测试可能没有抽到车辆极其密集的图片，只有跑完全部 12,248 张训练图片，才能检查最坏情况下的显存、标注和匹配计算。

完整 1 轮需要验证：

- 所有训练图片和标注都能读取；
- 密集目标 batch 不会 OOM；
- workers 长时运行不会退出或卡死；
- loss 为有限数值，不出现 `nan` / `inf`；
- 前向、反向、EMA、checkpoint 和验证全链路正常；
- 显存、温度和迭代时间能长时保持稳定。

执行：

```powershell
python tools\train.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  --cfg-options `
  train_cfg.max_epochs=1 `
  work_dir=./work_dirs/l315_full_1epoch_test
```

成功标志是完成整轮训练、保存 `epoch_1.pth` 并完成验证。迭代次数由最终 batch 决定：

```text
训练 batch 1：12248 次/epoch
训练 batch 2： 6124 次/epoch
训练 batch 4： 3062 次/epoch

验证 batch 1：3065 次
验证 batch 2：1533 次
验证 batch 4： 767 次
```

模考通过标准：

| 检查项 | 通过标准 | 失败处理 |
|---|---|---|
| GPU 显存 | 完整 1 轮无 OOM | 降低物理 batch，同时提高梯度累积 |
| loss | 无 `nan` / `inf`，整体有下降趋势 | 停止训练并检查标注、学习率和数值异常 |
| 数据读取 | 训练和验证均能跑完 | 定位报错图片或标注文件 |
| checkpoint | 生成 `epoch_1.pth` 和 `last_checkpoint` | 检查工作目录和磁盘空间 |
| 验证 | 输出四类 AP 和 mAP | 检查类别顺序、数据路径和评价器 |
| 长时稳定性 | 无 worker 崩溃、驱动重置或异常降频 | 减少 workers，检查电源和散热 |

> [!NOTE]
> 模考的 mAP 只用于确认模型已经开始学习，不代表最终精度。如果这 1 轮使用的就是最终固定配置，则 `epoch_1.pth` 可直接作为正式 72 轮训练的第 1 轮，无需重新开始。

当前已验证的第 1 轮结果：

```text
训练迭代：3062
显存峰值：约 5.5 GB（MMEngine 统计）
mAP: 0.3963
```

## 10. 🚀 继续正式 72 轮基线训练

第 1 轮已经是正式基线的一部分。使用同一工作目录从 `epoch_1.pth` 继续：

```powershell
python tools\train.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  --work-dir ./work_dirs/l315_full_1epoch_test `
  --resume
```

正确恢复时会看到：

```text
resumed epoch: 1
Epoch(train) [2][.../3062]
```

正式训练开始后，不再修改模型、数据、batch size、梯度累积、优化器、学习率、数据增强和输入尺寸。

## 11. ⏯️ 中断与恢复

配置每轮都保存 checkpoint。需要暂停时，尽量等待：

```text
Saving checkpoint at N epochs
```

然后按一次 `Ctrl+C`。恢复时重新执行第 10 节的 `--resume` 命令。

如果在某轮中间停止，当前未完成轮次会丢失，但可从上一个完整 checkpoint 继续。

## 12. 📊 训练产物和基线记录

权重、日志和配置快照保存在：

```text
D:\Object_Detection\O2-RT-DETR-main\work_dirs\l315_full_1epoch_test\
```

每 6 轮记录：

```text
epoch
mAP
Bus AP
Car AP
Truck AP
Van AP
```

已得到的基线节点：

| Epoch | Bus AP | Car AP | Truck AP | Van AP | mAP |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3499 | 0.8472 | 0.2935 | 0.0946 | 0.3963 |
| 6 | 0.8722 | 0.9046 | 0.7371 | 0.4461 | 0.7400 |

先完成 72 轮并保留完整学习曲线，再根据收敛情况统一确定后续基线、改进模型和消融实验的正式训练轮数。


