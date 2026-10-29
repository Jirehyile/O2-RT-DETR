# Copyright (c) ai4rs. All rights reserved.
from .dota_metric import DOTAMetric
from .rotated_coco_metric import RotatedCocoMetric
from .icdar2015_metric import ICDAR2015Metric
from .fair_metric import FAIRMetric

# COCOevalMP is unavailable in the MMDetection version required by this
# project. Keep the optional SARDet-100K metric from blocking unrelated DOTA
# datasets and models such as O2-RTDETR.
try:
    from .coco_metric_sardet_100k import CocoMetricSARDet100k
except ImportError:
    CocoMetricSARDet100k = None

__all__ = ['DOTAMetric', 'RotatedCocoMetric', 'ICDAR2015Metric', 'FAIRMetric']
if CocoMetricSARDet100k is not None:
    __all__.append('CocoMetricSARDet100k')
