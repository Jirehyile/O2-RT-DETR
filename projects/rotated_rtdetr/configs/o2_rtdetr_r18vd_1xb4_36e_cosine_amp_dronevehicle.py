"""O2-RTDETR-R18 在 DroneVehicle RGB 子集上的基线配置。

数据集使用去除四周白边后的 640×512 RGB 图像及 DOTA 格式旋转框标注。
输入图像保持长宽比缩放并填充至 1024×1024，以保留 O2-RT-DETR 的原始
输入设置，并避免直接拉伸导致车辆形状和旋转角度失真。

配置沿用已经验证稳定的单卡 8 GB 训练方案：单卡批量大小为 4，梯度累积
2 次，有效批量大小为 8；使用 AMP、2000 次迭代线性预热和 36 轮余弦退火。
"""

from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_dota import *


# DroneVehicle 的单模态 RGB 数据集。
dataset_root = 'D:/Object_Detection/data/dronevehicle_png/'
metainfo = dict(
    classes=('car', 'truck', 'bus', 'van', 'freight_car'),
    palette=[
        (0, 128, 255),
        (255, 165, 0),
        (220, 20, 60),
        (0, 180, 90),
        (148, 0, 211),
    ])

# DroneVehicle 包含 5 个车辆类别。
model['bbox_head']['num_classes'] = 5

# 所有车辆都可能以任意角度出现，不设置只能水平放置的类别。
train_pipeline[5]['rect_obj_labels'] = None

# 训练集：17,990 张 RGB 图像。
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

# 验证集：1,469 张 RGB 图像，每轮训练结束后用于选择最佳权重。
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

# 测试集：8,980 张带标注的 RGB 图像，仅用于最终模型评估。
test_dataloader = deepcopy(val_dataloader)
test_dataloader.update(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='test/labelTxt/',
        data_prefix=dict(img_path='test/images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
test_evaluator = deepcopy(val_evaluator)

# 训练36轮，每轮验证一次，以便根据验证集mAP选取最佳模型权重。
max_epochs = 36
train_cfg.update(
    max_epochs=max_epochs,
    val_interval=1)

# 先进行2000次迭代的线性预热，再在完整的36轮中使用余弦退火。
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=2000),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        eta_min_ratio=0.01)
]

# 单卡批量大小为4，累积2步后有效批量大小为8，与论文设置一致。
# AMP采用已经在当前设备上验证稳定的动态损失缩放参数。
optim_wrapper.update(
    type='AmpOptimWrapper',
    accumulative_counts=2,
    loss_scale=dict(
        init_scale=512.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100000))
auto_scale_lr.update(enable=False, base_batch_size=8)

# Windows 数据加载进程使用 spawn 启动方式。
env_cfg['mp_cfg']['mp_start_method'] = 'spawn'

# 与L315基线和后续创新实验分别保存，避免相互覆盖。
work_dir = './work_dirs/dronevehicle_o2_rtdetr_r18_36e_cosine_amp_b4_baseline'
