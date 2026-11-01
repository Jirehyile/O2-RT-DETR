<h1 align="center">O²-RTDETR × L315_vehicle</h1>

<p align="center"><strong>Windows 单卡环境下的遥感旋转车辆检测基线复现与创新实验工程</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.0.1">
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white" alt="CUDA 11.8">
  <img src="https://img.shields.io/badge/MMDetection-3.0.0-5C6BC0" alt="MMDetection 3.0.0">
  <img src="https://img.shields.io/badge/Test_mAP-83.07%25-success" alt="Test mAP 83.07%">
</p>

本仓库记录 O²-RTDETR-R18 在四类 L315_vehicle 数据集上的完整基线流程，包括环境搭建、硬件适配、正式训练、断点恢复、验证集选权重、测试集评估、可视化、预测结果导出、混淆矩阵和后续创新方向。

> [!NOTE]
> 原论文与官方项目介绍保存在 [README_OFFICIAL.md](README_OFFICIAL.md)。本 README 只描述当前仓库的 L315 基线复现和实验流程。

## 🧭 当前进度

- [x] 环境与依赖配置
- [x] L315 四分类数据加载
- [x] 单卡 8 GB 硬件适配
- [x] 114 轮收敛探索
- [x] 验证集选择最佳权重
- [x] 测试集正式评估
- [x] 测试集可视化
- [x] 预测结果导出
- [x] 混淆矩阵与误差分析
- [ ] 主创新实现与消融实验

```mermaid
flowchart LR
    A[环境与数据] --> B[硬件适配]
    B --> C[完整1轮压力测试]
    C --> D[正式训练]
    D --> E[验证集选择权重]
    E --> F[测试集最终评估]
    F --> G[可视化与混淆分析]
    G --> H[创新与消融实验]
```

## 📊 基线结论

### 最终配置

| 项目 | 设置 |
|---|---|
| 模型 | O²-RTDETR-R18-vd |
| 数据集 | L315_vehicle |
| 类别 | Bus、Car、Truck、Van |
| 输入尺寸 | 1024 × 1024 |
| 训练精度 | FP32，不启用 AMP |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU 8 GB |
| 训练 batch size | 4 |
| 验证/测试 batch size | 4 |
| 梯度累积 | 2 |
| 有效 batch size | 8 |
| num_workers | 2 |
| 优化器 | AdamW |
| 基础学习率 | 1e-4 |
| 验证间隔 | 每 6 轮 |
| 收敛探索 | 114 轮 |
| 推荐正式训练上限 | 90 轮，验证集选择最佳权重 |
| 最佳检查点 | epoch_78.pth |

### 正式结果

| 数据划分 | 权重 | mAP | AP50 |
|---|---|---:|---:|
| 验证集 | epoch_78.pth | 85.09% | 85.10% |
| 测试集 | epoch_78.pth | 83.07% | 83.10% |

测试集分类结果：

| 类别 | GT | Recall | AP |
|---|---:|---:|---:|
| Bus | 287 | 99.30% | 86.43% |
| Car | 23,059 | 99.67% | 90.19% |
| Truck | 3,654 | 99.37% | 85.65% |
| Van | 1,078 | 99.81% | 70.03% |

### 时间统计

| 项目 | 时间 |
|---|---:|
| 114 轮总运行时间 | 70 小时 25 分 24 秒 |
| 平均每轮（包含验证） | 约 37 分 04 秒 |
| 平均每轮（不包含验证） | 约 36 分 26 秒 |

## 📁 目录结构

```text
Object_Detection/
├── O2-RT-DETR-main/
│   ├── ai4rs/
│   ├── tools/
│   ├── projects/rotated_rtdetr/
│   │   └── configs/o2_rtdetr_r18vd_1xb1_72e_l315.py
│   ├── work_dirs/                         # 本地生成，不进入 Git
│   │   └── l315_o2_rtdetr_r18_baseline/
│   │       ├── epoch_78.pth
│   │       ├── test_predictions.pkl
│   │       ├── test_visualizations/
│   │       └── confusion_matrix/
│   └── README_OFFICIAL.md
└── data/L315_vehicle/
    ├── train/
    │   ├── images/
    │   └── labelTxt/
    ├── val/
    │   ├── images/
    │   └── labelTxt/
    └── test/
        ├── images/
        └── labelTxt/
```

| 划分 | 图片数 | 用途 |
|---|---:|---|
| train | 12,248 | 模型训练 |
| val | 3,065 | 训练期间验证和选择权重 |
| test | 3,350 | 最终评估，不参与调参 |

## 🐍 环境搭建

### 1. 创建环境

```powershell
conda create -n pytorch201 python=3.9 -y
conda activate pytorch201
cd D:\Object_Detection\O2-RT-DETR-main
```

### 2. 安装 PyTorch 2.0.1 + CUDA 11.8

```powershell
python -m pip install torch==2.0.1 torchvision==0.15.2 `
  --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装基础依赖

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

### 4. 安装 MMCV 和项目

```powershell
python -m pip install mmcv==2.0.1 `
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

python -m pip install -e . --no-deps
```

### 5. 验证环境

```powershell
python -c "import torch,torchvision,numpy,cv2,mmcv,mmengine,mmdet,ai4rs; print('torch:',torch.__version__); print('torchvision:',torchvision.__version__); print('numpy:',numpy.__version__); print('opencv:',cv2.__version__); print('mmcv:',mmcv.__version__); print('mmengine:',mmengine.__version__); print('mmdet:',mmdet.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0))"

python -m pip check
```

参考环境：

```text
Python       3.9.25
PyTorch      2.0.1+cu118
Torchvision  0.15.2+cu118
NumPy        1.26.4
OpenCV       4.10.0
MMCV         2.0.1
MMEngine     0.10.7
MMDetection  3.0.0
```

## 🧩 仓库已集成的兼容修复

拉取本仓库后不需要重复修改以下内容：

- Python 3.9 类型注解兼容；
- `ConcatDataset` 导入路径适配；
- 可选 SARDet 指标隔离；
- L315 标注头 `imagesource`、`gsd` 过滤；
- O²-RTDETR Decoder 与 MMDetection 3.0.0 接口适配；
- Windows `spawn` 多进程配置；
- L315 四分类配置及旋转框数据加载。

这些调整用于运行兼容，不改变 O²-RTDETR 的核心网络结构和原始损失定义。

## ⚙️ 数据与配置

配置文件：

```text
projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_1xb1_72e_l315.py
```

默认数据路径：

```python
dataset_root = 'D:/Object_Detection/data/L315_vehicle/'
```

如果队友的数据集位置不同，只修改 `dataset_root`，不要改变类别顺序和 train/val/test 划分。

确认配置：

```powershell
python -c "from mmengine.config import Config; c=Config.fromfile(r'projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_1xb1_72e_l315.py'); print('train batch:',c.train_dataloader.batch_size); print('val batch:',c.val_dataloader.batch_size); print('test batch:',c.test_dataloader.batch_size); print('workers:',c.train_dataloader.num_workers); print('accumulation:',c.optim_wrapper.accumulative_counts); print('effective batch:',c.train_dataloader.batch_size*c.optim_wrapper.accumulative_counts); print('test ann:',c.test_dataloader.dataset.ann_file)"
```

## 🧪 新电脑硬件适配

不同电脑不能直接假设 `batch=4、workers=2` 一定可用。先做小规模冒烟测试，再进行完整一轮压力测试。

### 1. 小规模冒烟测试

```powershell
python tools\train.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  --cfg-options `
  train_dataloader.batch_size=1 `
  train_dataloader.num_workers=0 `
  train_dataloader.persistent_workers=False `
  train_dataloader.dataset.indices=32 `
  optim_wrapper.accumulative_counts=8 `
  train_cfg.max_epochs=1 `
  work_dir=./work_dirs/hardware_test_b1
```

逐级测试并保持有效 batch 为 8：

| 物理 batch | 梯度累积 | 有效 batch |
|---:|---:|---:|
| 1 | 8 | 8 |
| 2 | 4 | 8 |
| 4 | 2 | 8 |

出现 `CUDA out of memory` 时退回上一个稳定值。workers 建议从0开始测试，再尝试2；只有更快且稳定时才继续增加。

### 2. 完整一轮压力测试

```powershell
python tools\train.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  --cfg-options `
  train_cfg.max_epochs=1 `
  work_dir=./work_dirs/l315_full_epoch_test
```

确认全部12,248张训练图片可读取、无OOM、loss无`nan/inf`、checkpoint正常保存且验证完整结束。

> [!CAUTION]
> 当前基线保持 FP32。不要直接加入 `--amp`；本项目部分旋转匹配计算在当前依赖组合下不支持完整FP16路径。

## 🚀 正式训练

推荐最大训练90轮，每6轮验证一次，并在90轮以内选择验证集mAP最高的检查点。114轮仅用于本次收敛探索，不代表后续实验必须训练114轮。

```powershell
python tools\train.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  --cfg-options `
  train_cfg.max_epochs=90 `
  train_cfg.val_interval=6 `
  work_dir=./work_dirs/l315_o2_rtdetr_r18_baseline
```

### 中断与恢复

尽量在出现 `Saving checkpoint at N epochs` 后停止。恢复命令：

```powershell
python tools\train.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  --work-dir ./work_dirs/l315_o2_rtdetr_r18_baseline `
  --resume
```

中途修改模型、数据划分、batch、梯度累积、优化器或学习率后，不能继续沿用原实验作为同一基线。

## ✅ 测试集最终评估

训练期间通过验证集选择最佳权重。本次最佳权重为 `epoch_78.pth`：

```powershell
python tools\test.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  work_dirs\l315_o2_rtdetr_r18_baseline\epoch_78.pth
```

3350张测试图片、batch 4 对应 `Epoch(test) [838/838]`。创新阶段仍使用验证集筛选，测试集只用于最终报告。

## 🖼️ 可视化与预测导出

生成测试集可视化：

```powershell
python tools\test.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  work_dirs\l315_o2_rtdetr_r18_baseline\epoch_78.pth `
  --show-dir test_visualizations `
  --wait-time 0
```

每张图片左侧为真实旋转框，右侧为预测旋转框与置信度。

导出逐图预测结果：

```powershell
python tools\test.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  work_dirs\l315_o2_rtdetr_r18_baseline\epoch_78.pth `
  --out work_dirs\l315_o2_rtdetr_r18_baseline\test_predictions.pkl
```

`test_predictions.pkl`保存逐图预测类别、置信度和旋转框坐标。它不是模型权重，主要用于离线分析。

## 🔍 混淆矩阵

```powershell
python tools\analysis_tools\confusion_matrix.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  work_dirs\l315_o2_rtdetr_r18_baseline\test_predictions.pkl `
  work_dirs\l315_o2_rtdetr_r18_baseline\confusion_matrix `
  --score-thr 0.3 `
  --tp-iou-thr 0.5
```

主要结论：

1. Van AP为70.03%，显著低于其他类别；
2. Van→Car为23.0%，Van→Truck为6.5%；
3. Car有106,534个训练实例，Van只有2,465个，相差约43倍；
4. 背景误检中80.5%被预测为Car；
5. 单纯延长训练到114轮不能继续稳定提升性能。

## 🧠 后续研究方向

**主创新：类别均衡原型引导查询增强（BCP-QE）。** 在Decoder查询和分类头之间建立类别原型交互，增强Van、Car和Truck之间的细粒度判别，并采用类别均衡更新避免Car主导特征空间。

**补充创新一：混淆感知原型间隔损失（CAPM Loss）。** 根据Van→Car和Van→Truck的混淆关系设置针对性特征间隔。

**补充创新二：背景难负查询抑制（BHQS）。** 筛选高置信度背景查询，重点降低Background→Car误检。

正式编码前需要检索类别原型、DETR查询增强、长尾检测和遥感细粒度车辆检测相关论文，确认具体实现与已有方法不存在高度重复。

## 🤝 双人协作与权重共享

`.gitignore`默认忽略`work_dirs`，因此GitHub不会包含权重、日志、预测文件和可视化图片。`epoch_78.pth`约414MB，也超过GitHub普通文件100MB限制。

队友拉取代码后创建目录：

```powershell
New-Item -ItemType Directory -Force `
  work_dirs\l315_o2_rtdetr_r18_baseline
```

通过网盘、U盘或局域网共享：

```text
必需：epoch_78.pth
可选：test_predictions.pkl
可选：基线汇总Excel和典型可视化图片
```

将权重放到：

```text
work_dirs/l315_o2_rtdetr_r18_baseline/epoch_78.pth
```

基线权重用于复现与对比。创新模型应使用与基线相同的初始化重新训练，不应默认从`epoch_78.pth`继续微调。

## 📦 重要产物

| 文件 | 作用 | 是否必须共享 |
|---|---|---|
| `epoch_78.pth` | 最佳基线模型权重 | 是 |
| `test_predictions.pkl` | 测试集逐图结构化预测 | 按需 |
| `test_visualizations/` | 真实框与预测框对照图 | 按需 |
| `confusion_matrix/` | 混淆矩阵和误差分析 | 按需 |
| `baseline_validation_results_1_114.csv` | 验证集逐轮结果 | 建议 |
| 基线汇总Excel | 配置、训练、验证、测试、混淆与研究方向汇总 | 建议 |

## 📚 引用与官方资料

论文、官方模型说明和引用格式请查看 [README_OFFICIAL.md](README_OFFICIAL.md)。
