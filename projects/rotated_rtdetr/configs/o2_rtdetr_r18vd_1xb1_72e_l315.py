"""O2-RTDETR-R18 configuration for the four-class L315 vehicle dataset.

This keeps the paper's 1024x1024 input and optimizer settings while adapting
the dataloaders to a single 8 GB GPU on Windows. Gradient accumulation keeps
the effective batch size at eight images.
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

# DOTA uses special labels for objects that are always horizontal. L315's
# vehicles have arbitrary orientations, so retain unrestricted random angles.
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

# Use the validation split for the first end-to-end evaluation. The supplied
# test split can be wired in after the training and validation path is proven.
test_dataloader = deepcopy(val_dataloader)
test_evaluator = deepcopy(val_evaluator)

# Paper effective batch size: 2 GPUs x 4 images = 8.
# On one GPU, use batch size 4 with two-step gradient accumulation.
optim_wrapper.update(
    type='OptimWrapper',
    accumulative_counts=2)
auto_scale_lr.update(enable=False, base_batch_size=8)

# ``fork`` is unavailable on Windows. Workers are disabled for the initial
# run, but using ``spawn`` also keeps later worker increases portable.
env_cfg['mp_cfg']['mp_start_method'] = 'spawn'

work_dir = './work_dirs/o2_rtdetr_r18vd_1xb1_72e_l315'
