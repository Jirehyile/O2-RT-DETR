# O²-RTDETR 在 L315_vehicle 上的基线训练流程

本文档记录从拿到 `D:\Object_Detection` 工程后，到建立 Python 环境、适配源码、配置 `L315_vehicle` 数据集，再到启动正式基线训练的必要步骤。

原项目的论文、模型介绍与官方说明见 [README_OFFICIAL.md](README_OFFICIAL.md)。

## 1. 最终基线设置

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

## 2. 目录结构

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

## 3. 创建 Conda 环境

在 PowerShell 中执行：

```powershell
conda create -n pytorch201 python=3.9 -y
conda activate pytorch201
cd D:\Object_Detection\O2-RT-DETR-main
```

后续所有命令都在 `pytorch201` 环境和项目根目录下执行。

## 4. 安装依赖

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

## 5. 必要的源码兼容修复

下列修复用于使项目适配 Python 3.9 + MMDetection 3.0.0，不改变 O²-RTDETR 的网络结构。

### 5.1 Python 3.9 类型注解兼容

文件：

```text
projects\rotated_rtdetr\rotated_rtdetr\prob_iou.py
```

在第一行加入：

```python
from __future__ import annotations
```

### 5.2 ConcatDataset 导入路径

文件：

```text
ai4rs\datasets\dataset_wrappers.py
```

使用：

```python
from mmengine.dataset import ConcatDataset as MMDET_ConcatDataset
```

### 5.3 避免可选 SARDet 评价器阻断 DOTA 评价

文件：

```text
ai4rs\evaluation\metrics\__init__.py
```

内容保持为：

```python
from .dota_metric import DOTAMetric
from .rotated_coco_metric import RotatedCocoMetric
from .icdar2015_metric import ICDAR2015Metric
from .fair_metric import FAIRMetric

try:
    from .coco_metric_sardet_100k import CocoMetricSARDet100k
except ImportError:
    CocoMetricSARDet100k = None

__all__ = ['DOTAMetric', 'RotatedCocoMetric', 'ICDAR2015Metric', 'FAIRMetric']
if CocoMetricSARDet100k is not None:
    __all__.append('CocoMetricSARDet100k')
```

### 5.4 忽略 L315 DOTA 格式中的元数据行

文件：

```text
ai4rs\datasets\dota.py
```

在 `bbox_info = si.split()` 后增加：

```python
if len(bbox_info) < 10:
    continue
```

用于忽略：

```text
imagesource:lab315
gsd:NULL
```

### 5.5 补齐项目 Decoder 与 MMDetection 3.0.0 的接口

文件：

```text
projects\rotated_rtdetr\rotated_rtdetr\rotated_rtdetr.py
```

在 `pre_decoder()` 之后、`gen_encoder_output_proposals()` 之前加入：

```python
    def forward_decoder(
            self,
            query: Tensor,
            memory: Tensor,
            memory_mask: Tensor,
            reference_points: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            valid_ratios: Tensor,
            dn_mask: Optional[Tensor] = None,
            cls_branches: nn.ModuleList = None,
            eval_idx: int = -1) -> Dict:

        outputs_classes, outputs_coords = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            cls_branches=cls_branches,
            eval_idx=eval_idx)

        return dict(
            hidden_states=outputs_classes,
            references=outputs_coords)
```

## 6. L315_vehicle 基线配置

配置文件：

```text
projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py
```

关键配置必须保持为：

```python
from copy import deepcopy
from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_dota import *

dataset_root = 'D:/Object_Detection/data/L315_vehicle/'
metainfo = dict(
    classes=('Bus', 'Car', 'Truck', 'Van'),
    palette=[(220, 20, 60), (0, 128, 255),
             (255, 165, 0), (0, 180, 90)])

model['bbox_head']['num_classes'] = 4
train_pipeline[5]['rect_obj_labels'] = None

train_dataloader.update(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader.update(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='val/labelTxt/',
        data_prefix=dict(img_path='val/images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = deepcopy(val_dataloader)
test_evaluator = deepcopy(val_evaluator)

optim_wrapper.update(
    type='OptimWrapper',
    accumulative_counts=2)
auto_scale_lr.update(enable=False, base_batch_size=8)

env_cfg['mp_cfg']['mp_start_method'] = 'spawn'

work_dir = './work_dirs/o2_rtdetr_r18vd_1xb1_72e_l315'
```

不要使用 `--amp`。该项目的 KLD 匹配代价在当前 PyTorch 版本下不支持 FP16 矩阵行列式运算。

## 7. 检查最终配置

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

## 8. 完整数据集 1 轮压力测试

在正式长时间训练前，先用全部训练集跑 1 轮，确认密集目标图片不会导致显存溢出：

```powershell
python tools\train.py `
  projects\rotated_rtdetr\configs\o2_rtdetr_r18vd_1xb1_72e_l315.py `
  --cfg-options `
  train_cfg.max_epochs=1 `
  work_dir=./work_dirs/l315_full_1epoch_test
```

成功标志：

```text
Saving checkpoint at 1 epochs
Epoch(val) [1][767/767]
```

当前已验证的第 1 轮结果：

```text
训练迭代：3062
显存峰值：约 5.5 GB（MMEngine 统计）
mAP: 0.3963
```

## 9. 继续正式 72 轮基线训练

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

## 10. 中断与恢复

配置每轮都保存 checkpoint。需要暂停时，尽量等待：

```text
Saving checkpoint at N epochs
```

然后按一次 `Ctrl+C`。恢复时重新执行第 9 节的 `--resume` 命令。

如果在某轮中间停止，当前未完成轮次会丢失，但可从上一个完整 checkpoint 继续。

## 11. 训练产物和基线记录

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


