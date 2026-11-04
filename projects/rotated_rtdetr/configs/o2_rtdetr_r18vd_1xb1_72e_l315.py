"""面向 L315 四类别车辆数据集的 O2-RTDETR-R18 配置。

该配置保留论文使用的 1024×1024 输入尺寸和优化器设置，同时将数据加载器
调整为适用于 Windows 系统上的单张 8 GB 显存 GPU。通过梯度累积将有效
批量大小保持为 8 张图像。
"""

from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_dota import *


dataset_root = 'D:/Object_Detection/data/L315_vehicle/'
metainfo = dict(
    classes=('Bus', 'Car', 'Truck', 'Van'),
    palette=[(220, 20, 60), (0, 128, 255), (255, 165, 0), (0, 180, 90)])

model['bbox_head']['num_classes'] = 4

# DOTA 会为始终保持水平的目标使用特殊标签。L315 中的车辆具有任意朝向，
# 因此这里不限制随机旋转角度。
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

# 使用带标注的 L315 测试集评估最终选定的模型权重。
# 训练和验证流程确认无误后，即可使用该测试集配置进行最终测试。
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

# 论文的有效批量大小为：2 张 GPU × 每张 GPU 4 张图像 = 8。
# 在单张 GPU 上使用批量大小 4，并进行 2 步梯度累积。
optim_wrapper.update(
    type='OptimWrapper',
    accumulative_counts=2)
auto_scale_lr.update(enable=False, base_batch_size=8)

# Windows 不支持 ``fork`` 启动方式，因此使用 ``spawn``，同时保证后续增加
# 数据加载进程数量时仍能在 Windows 上正常运行。
env_cfg['mp_cfg']['mp_start_method'] = 'spawn'

work_dir = './work_dirs/l315_o2_rtdetr_r18_baseline'
