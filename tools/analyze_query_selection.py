"""Diagnose encoder Top-K query selection without changing the model.

This script attaches a runtime wrapper to ``pre_decoder`` and observes the
encoder proposal pool used by O2-RTDETR.  The original ``pre_decoder`` is
still called unchanged, so model predictions, configuration and checkpoints
are not modified.

The diagnosis reports:

1. class-agnostic and class-aware encoder Recall@K;
2. Top-K predicted-class composition;
3. valid, background and duplicate candidate rates;
4. the same measurements for sparse/medium/dense images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import os.path as osp
import pickle
from collections import defaultdict
from types import MethodType
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.utils import register_all_modules as register_all_modules_mmdet

from ai4rs.structures.bbox import rbbox_overlaps
from ai4rs.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose O2-RTDETR encoder Top-K query selection')
    parser.add_argument('config', help='model config file')
    parser.add_argument('checkpoint', help='trained checkpoint')
    parser.add_argument(
        '--split', choices=('val', 'test'), default='val',
        help='dataset split used for diagnosis (default: val)')
    parser.add_argument(
        '--topk', type=int, nargs='+', default=[100, 200, 300, 400],
        help='candidate budgets to diagnose')
    parser.add_argument(
        '--iou-thr', type=float, default=0.5,
        help='rotated IoU threshold for candidate/GT coverage')
    parser.add_argument(
        '--output-dir', default='work_dirs/dronevehicle_query_diagnosis',
        help='directory for CSV, JSON and PKL outputs')
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='only diagnose the first N images (useful for a smoke test)')
    parser.add_argument(
        '--no-raw', action='store_true',
        help='do not save per-image candidates to query_candidates.pkl')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config values, e.g. test_dataloader.num_workers=0')
    return parser.parse_args()


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def tensor_from_boxes(boxes) -> torch.Tensor:
    if hasattr(boxes, 'tensor'):
        return boxes.tensor
    return boxes


def write_csv(path: str, fieldnames: Sequence[str], rows: Iterable[dict]):
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


class QueryDiagnosisCollector:
    """Collect and summarize encoder candidates produced before decoding."""

    def __init__(self, model, topk: Sequence[int], iou_thr: float,
                 class_names: Sequence[str], keep_raw: bool = True):
        self.model = model
        self.topk = sorted(set(int(k) for k in topk if int(k) > 0))
        self.iou_thr = float(iou_thr)
        self.class_names = list(class_names)
        self.keep_raw = keep_raw
        self.records: List[dict] = []
        self._original_pre_decoder = model.pre_decoder

    def install(self):
        collector = self
        original = self._original_pre_decoder

        def wrapped_pre_decoder(this, memory, memory_mask, spatial_shapes,
                                batch_data_samples=None):
            # Reproduce only the proposal calculations used immediately before
            # Top-K.  No tensor below is fed back to the model.
            with torch.no_grad():
                output_memory, output_proposals = \
                    this.gen_encoder_output_proposals(
                        memory, memory_mask, spatial_shapes)
                cls_branch = this.bbox_head.cls_branches[
                    this.decoder.num_layers]
                reg_branch = this.bbox_head.reg_branches[
                    this.decoder.num_layers]
                encoder_logits = cls_branch(output_memory)
                encoder_coords = (
                    reg_branch(output_memory) + output_proposals).sigmoid()
                collector.consume(
                    encoder_logits, encoder_coords, batch_data_samples)

            # The actual model follows its untouched original path.
            return original(
                memory=memory,
                memory_mask=memory_mask,
                spatial_shapes=spatial_shapes,
                batch_data_samples=batch_data_samples)

        self.model.pre_decoder = MethodType(wrapped_pre_decoder, self.model)

    def uninstall(self):
        self.model.pre_decoder = self._original_pre_decoder

    def consume(self, logits: torch.Tensor, coords: torch.Tensor,
                data_samples):
        pool_size = logits.shape[1]
        max_k = min(max(self.topk), pool_size)
        # O2-RTDETR uses raw maximum class logits for Top-K selection.
        ranking_scores, ranking_labels = logits.max(dim=-1)
        indices = ranking_scores.topk(max_k, dim=1).indices

        for batch_index, sample in enumerate(data_samples):
            image_logits = logits[batch_index]
            image_scores = image_logits.sigmoid()
            selected = indices[batch_index]
            selected_coords = coords[batch_index, selected]
            selected_probs, selected_labels = image_scores[selected].max(-1)

            img_h, img_w = sample.metainfo['img_shape'][:2]
            angle_factor = float(self.model.bbox_head.angle_factor)
            factor = selected_coords.new_tensor(
                [img_w, img_h, img_w, img_h, angle_factor])
            selected_boxes = selected_coords * factor

            # The validation pipeline deliberately loads annotations *after*
            # Resize, so GT boxes remain in original-image coordinates (this
            # is also what DOTAMetric expects). Encoder proposals, however,
            # are expressed in resized-image coordinates. Rescale proposals
            # back before computing IoU; otherwise apparently correct boxes
            # are shifted/scaled and Recall@K becomes close to zero.
            scale_factor = sample.metainfo.get('scale_factor', (1.0, 1.0))
            scale_x, scale_y = float(scale_factor[0]), float(scale_factor[1])
            selected_boxes[:, 0] /= scale_x
            selected_boxes[:, 1] /= scale_y
            selected_boxes[:, 2] /= scale_x
            selected_boxes[:, 3] /= scale_y

            gt_instances = sample.gt_instances
            gt_boxes = tensor_from_boxes(gt_instances.bboxes).to(
                device=selected_boxes.device, dtype=selected_boxes.dtype)
            gt_labels = gt_instances.labels.to(selected_boxes.device)

            if len(gt_boxes) > 0:
                ious = rbbox_overlaps(selected_boxes, gt_boxes)
            else:
                ious = selected_boxes.new_zeros((max_k, 0))

            image_record = {
                'img_id': str(sample.metainfo.get(
                    'img_id', sample.metainfo.get('img_path', len(self.records)))),
                'img_path': sample.metainfo.get('img_path', ''),
                'num_gt': int(len(gt_boxes)),
                'gt_labels': gt_labels.detach().cpu().numpy().astype(np.int16),
                'metrics': {},
            }

            for requested_k in self.topk:
                k = min(requested_k, max_k)
                k_ious = ious[:k]
                k_labels = selected_labels[:k]
                k_scores = selected_probs[:k]

                if len(gt_boxes) > 0:
                    best_iou_per_candidate, best_gt_per_candidate = \
                        k_ious.max(dim=1)
                    best_iou_per_gt, _ = k_ious.max(dim=0)
                    valid = best_iou_per_candidate >= self.iou_thr
                    covered = best_iou_per_gt >= self.iou_thr

                    class_match = (k_labels[:, None] == gt_labels[None, :])
                    class_aware_ious = k_ious.masked_fill(~class_match, -1)
                    best_class_iou_per_gt, _ = class_aware_ious.max(dim=0)
                    class_covered = best_class_iou_per_gt >= self.iou_thr

                    assigned_gt = best_gt_per_candidate[valid]
                    if assigned_gt.numel() > 0:
                        counts = torch.bincount(
                            assigned_gt, minlength=len(gt_boxes))
                        duplicate_count = int(
                            (counts - 1).clamp(min=0).sum().item())
                    else:
                        duplicate_count = 0
                else:
                    valid = torch.zeros(k, dtype=torch.bool,
                                        device=selected_boxes.device)
                    covered = torch.zeros(0, dtype=torch.bool,
                                           device=selected_boxes.device)
                    class_covered = covered
                    duplicate_count = 0

                image_record['metrics'][requested_k] = {
                    'actual_k': k,
                    'valid_count': int(valid.sum().item()),
                    'background_count': int(k - valid.sum().item()),
                    'duplicate_count': duplicate_count,
                    'covered': covered.detach().cpu().numpy().astype(np.uint8),
                    'class_covered': class_covered.detach().cpu().numpy().astype(
                        np.uint8),
                    'predicted_counts': np.bincount(
                        k_labels.detach().cpu().numpy(),
                        minlength=len(self.class_names)).astype(np.int32),
                    'mean_score': float(k_scores.mean().item()),
                }

            if self.keep_raw:
                image_record['top_candidates'] = {
                    'indices': selected.detach().cpu().numpy().astype(np.int32),
                    'scores': selected_probs.detach().cpu().numpy().astype(
                        np.float32),
                    'labels': selected_labels.detach().cpu().numpy().astype(
                        np.int16),
                    'bboxes': selected_boxes.detach().cpu().numpy().astype(
                        np.float32),
                }
            self.records.append(image_record)

    def summarize(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        gt_counts = [record['num_gt'] for record in self.records]
        q33 = percentile(gt_counts, 33.333)
        q67 = percentile(gt_counts, 66.667)

        def density_group(num_gt):
            if num_gt <= q33:
                return 'sparse'
            if num_gt <= q67:
                return 'medium'
            return 'dense'

        recall_rows = []
        composition_rows = []
        image_rows = []
        density_rows = []

        for k in self.topk:
            for class_id, class_name in enumerate(self.class_names):
                total_gt = 0
                covered_gt = 0
                class_covered_gt = 0
                predicted_candidates = 0
                total_candidates = 0
                for record in self.records:
                    mask = record['gt_labels'] == class_id
                    metrics = record['metrics'][k]
                    total_gt += int(mask.sum())
                    covered_gt += int(metrics['covered'][mask].sum())
                    class_covered_gt += int(
                        metrics['class_covered'][mask].sum())
                    predicted_candidates += int(
                        metrics['predicted_counts'][class_id])
                    total_candidates += metrics['actual_k']
                recall_rows.append({
                    'K': k,
                    'class_id': class_id,
                    'class': class_name,
                    'gt_count': total_gt,
                    'class_agnostic_recall': (
                        covered_gt / total_gt if total_gt else ''),
                    'class_aware_recall': (
                        class_covered_gt / total_gt if total_gt else ''),
                    'recall_gap': (
                        (covered_gt - class_covered_gt) / total_gt
                        if total_gt else ''),
                })
                composition_rows.append({
                    'K': k,
                    'class_id': class_id,
                    'class': class_name,
                    'predicted_candidates': predicted_candidates,
                    'candidate_ratio': (
                        predicted_candidates / total_candidates
                        if total_candidates else ''),
                })

            for record in self.records:
                metrics = record['metrics'][k]
                actual_k = metrics['actual_k']
                image_rows.append({
                    'img_id': record['img_id'],
                    'img_path': record['img_path'],
                    'density_group': density_group(record['num_gt']),
                    'num_gt': record['num_gt'],
                    'K': k,
                    'valid_rate': metrics['valid_count'] / actual_k,
                    'background_rate': metrics['background_count'] / actual_k,
                    'duplicate_rate': metrics['duplicate_count'] / actual_k,
                    'gt_coverage': (
                        float(metrics['covered'].mean())
                        if record['num_gt'] else ''),
                    'class_aware_gt_coverage': (
                        float(metrics['class_covered'].mean())
                        if record['num_gt'] else ''),
                    'mean_candidate_score': metrics['mean_score'],
                })

            for group_name in ('sparse', 'medium', 'dense'):
                group_records = [
                    r for r in self.records
                    if density_group(r['num_gt']) == group_name
                ]
                for class_id, class_name in enumerate(self.class_names):
                    total_gt = covered_gt = class_covered_gt = 0
                    valid = background = duplicate = candidates = 0
                    for record in group_records:
                        mask = record['gt_labels'] == class_id
                        metrics = record['metrics'][k]
                        total_gt += int(mask.sum())
                        covered_gt += int(metrics['covered'][mask].sum())
                        class_covered_gt += int(
                            metrics['class_covered'][mask].sum())
                        valid += metrics['valid_count']
                        background += metrics['background_count']
                        duplicate += metrics['duplicate_count']
                        candidates += metrics['actual_k']
                    density_rows.append({
                        'density_group': group_name,
                        'K': k,
                        'class_id': class_id,
                        'class': class_name,
                        'images': len(group_records),
                        'gt_count': total_gt,
                        'class_agnostic_recall': (
                            covered_gt / total_gt if total_gt else ''),
                        'class_aware_recall': (
                            class_covered_gt / total_gt if total_gt else ''),
                        'valid_candidate_rate_all_classes': (
                            valid / candidates if candidates else ''),
                        'background_rate_all_classes': (
                            background / candidates if candidates else ''),
                        'duplicate_rate_all_classes': (
                            duplicate / candidates if candidates else ''),
                    })

        write_csv(
            osp.join(output_dir, 'encoder_recall_at_k.csv'),
            ['K', 'class_id', 'class', 'gt_count',
             'class_agnostic_recall', 'class_aware_recall', 'recall_gap'],
            recall_rows)
        write_csv(
            osp.join(output_dir, 'topk_class_distribution.csv'),
            ['K', 'class_id', 'class', 'predicted_candidates',
             'candidate_ratio'], composition_rows)
        write_csv(
            osp.join(output_dir, 'candidate_analysis_per_image.csv'),
            ['img_id', 'img_path', 'density_group', 'num_gt', 'K',
             'valid_rate', 'background_rate', 'duplicate_rate', 'gt_coverage',
             'class_aware_gt_coverage', 'mean_candidate_score'], image_rows)
        write_csv(
            osp.join(output_dir, 'density_group_analysis.csv'),
            ['density_group', 'K', 'class_id', 'class', 'images', 'gt_count',
             'class_agnostic_recall', 'class_aware_recall',
             'valid_candidate_rate_all_classes',
             'background_rate_all_classes',
             'duplicate_rate_all_classes'], density_rows)

        summary = {
            'images': len(self.records),
            'total_gt': int(sum(gt_counts)),
            'topk': self.topk,
            'iou_threshold': self.iou_thr,
            'density_thresholds': {
                'sparse_max_gt': q33,
                'medium_max_gt': q67,
            },
            'class_names': self.class_names,
            'interpretation': {
                'class_agnostic_recall':
                    'Whether any encoder candidate geometrically covers a GT.',
                'class_aware_recall':
                    'Whether a covering candidate also predicts the GT class.',
                'recall_gap':
                    'Large gap means classification, not geometric allocation, '
                    'is the main failure.',
                'duplicate_rate':
                    'Candidates beyond the first valid candidate assigned to '
                    'the same GT, divided by K.',
            },
        }
        with open(osp.join(output_dir, 'query_diagnosis_summary.json'), 'w',
                  encoding='utf-8') as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        if self.keep_raw:
            with open(osp.join(output_dir, 'query_candidates.pkl'), 'wb') as f:
                pickle.dump(self.records, f, protocol=pickle.HIGHEST_PROTOCOL)

        return summary


def main():
    args = parse_args()
    if not 0 < args.iou_thr <= 1:
        raise ValueError('--iou-thr must be in (0, 1].')

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = 'none'
    cfg.load_from = args.checkpoint
    cfg.resume = False
    cfg.work_dir = args.output_dir
    if args.split == 'val':
        cfg.test_dataloader = cfg.val_dataloader
    # Evaluation hooks/metrics are unnecessary; this is a diagnostic forward.

    runner = Runner.from_cfg(cfg)
    # Match Runner.test() lifecycle so checkpoint-related hooks (especially
    # EMAHook used by this project) are initialized before loading weights.
    runner.call_hook('before_run')
    runner.load_or_resume()
    runner.call_hook('before_test')
    runner.call_hook('before_test_epoch')
    model = unwrap_model(runner.model)
    model.eval()

    class_names = runner.test_dataloader.dataset.metainfo.get('classes', None)
    if not class_names:
        class_names = tuple(
            str(i) for i in range(model.bbox_head.num_classes))

    collector = QueryDiagnosisCollector(
        model=model,
        topk=args.topk,
        iou_thr=args.iou_thr,
        class_names=class_names,
        keep_raw=not args.no_raw)
    collector.install()

    processed = 0
    try:
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(runner.test_dataloader):
                model.test_step(data_batch)
                processed = len(collector.records)
                if batch_idx % 20 == 0:
                    print(f'[query diagnosis] processed images: {processed}')
                if args.max_images is not None and processed >= args.max_images:
                    # A batch may make the result slightly larger than N.
                    collector.records = collector.records[:args.max_images]
                    break
    finally:
        collector.uninstall()

    if not collector.records:
        raise RuntimeError('No images were processed.')
    summary = collector.summarize(args.output_dir)
    runner.call_hook('after_test_epoch', metrics={})
    runner.call_hook('after_test')
    runner.call_hook('after_run')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Diagnosis saved to: {osp.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()
