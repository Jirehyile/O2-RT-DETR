"""O2-RTDETR-R18 在 DroneVehicle RGB 子集上的 24 轮训练配置。

数据集使用去除四周白边后的 640×512 RGB 图像及 DOTA 格式旋转框标注。
输入图像保持长宽比缩放并填充至 1024×1024。

单卡批量大小为 4，梯度累积 2 次，有效批量大小为 8；
使用 AMP、2000 次迭代线性预热和 24 轮余弦退火。
"""

from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from projects.rotated_rtdetr.configs.o2_rtdetr_r18vd_2xb4_72e_dota import *


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

model['bbox_head']['num_classes'] = 5

# 所有车辆允许任意旋转角度
train_pipeline[5]['rect_obj_labels'] = None

# 训练集
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

# 验证集
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

# 测试集
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

# 训练 24 轮，每轮验证一次
max_epochs = 24

train_cfg.update(
    max_epochs=max_epochs,
    val_interval=1)

# 2000 次迭代线性预热 + 24 轮余弦退火
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

# 单卡 batch size=4，梯度累积 2 次，有效 batch size=8
optim_wrapper.update(
    type='AmpOptimWrapper',
    accumulative_counts=2,
    loss_scale=dict(
        init_scale=512.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100000))

auto_scale_lr.update(
    enable=False,
    base_batch_size=8)

# Windows 多进程采用 spawn
env_cfg['mp_cfg']['mp_start_method'] = 'spawn'

# 独立输出目录
work_dir = './work_dirs/dronevehicle_o2_rtdetr_r18_24e_cosine_amp_b4_baseline'