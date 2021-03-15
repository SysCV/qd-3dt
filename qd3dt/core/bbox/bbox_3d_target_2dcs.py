import torch

from .transforms import bbox2delta
from ..utils import multi_apply


def bbox_3d_target(pos_bboxes_list,
                   neg_bboxes_list,
                   pos_gt_bboxes_list,
                   pos_gt_labels_list,
                   pos_gt_depth_list,
                   pos_gt_alpha_list,
                   pos_gt_roty_list,
                   pos_gt_dim_list,
                   pos_gt_2dc_list,
                   cfg,
                   reg_classes=1,
                   target_means=[.0, .0, .0, .0],
                   target_stds=[1.0, 1.0, 1.0, 1.0],
                   concat=True):
    labels, label_weights, \
    bbox_targets, bbox_weights, \
    depth_targets, depth_weights, \
    alpha_targets, alpha_weights, \
    roty_targets, roty_weights, \
    dim_targets, dim_weights, \
    cen_2d_targets, cen_2d_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        pos_gt_depth_list,
        pos_gt_alpha_list,
        pos_gt_roty_list,
        pos_gt_dim_list,
        pos_gt_2dc_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
        depth_targets = torch.cat(depth_targets, 0)
        depth_weights = torch.cat(depth_weights, 0)
        alpha_targets = torch.cat(alpha_targets, 0)
        alpha_weights = torch.cat(alpha_weights, 0)
        roty_targets = torch.cat(roty_targets, 0)
        roty_weights = torch.cat(roty_weights, 0)
        dim_targets = torch.cat(dim_targets, 0)
        dim_weights = torch.cat(dim_weights, 0)
        cen_2d_targets = torch.cat(cen_2d_targets, 0)
        cen_2d_weights = torch.cat(cen_2d_weights, 0)

    return labels, label_weights, bbox_targets, bbox_weights, \
           depth_targets, depth_weights, alpha_targets, alpha_weights, \
           roty_targets, roty_weights, dim_targets, dim_weights, \
           cen_2d_targets, cen_2d_weights


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)

    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       pos_gt_depths,
                       pos_gt_alphas,
                       pos_gt_rotys,
                       pos_gt_dims,
                       pos_gt_2dc,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    depth_targets = pos_bboxes.new_zeros(num_samples, )
    depth_weights = pos_bboxes.new_zeros(num_samples, )
    alpha_targets = pos_bboxes.new_zeros(num_samples, )
    alpha_weights = pos_bboxes.new_zeros(num_samples, )
    roty_targets = pos_bboxes.new_zeros(num_samples, )
    roty_weights = pos_bboxes.new_zeros(num_samples, )
    dim_targets = pos_bboxes.new_zeros(num_samples, 3)
    dim_weights = pos_bboxes.new_zeros(num_samples, 3)
    cen_2d_targets = pos_bboxes.new_zeros(num_samples, 2)
    cen_2d_weights = pos_bboxes.new_zeros(num_samples, 2)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight

        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        if pos_bbox_targets is not None:
            bbox_targets[:num_pos, :] = pos_bbox_targets[:num_pos]
            bbox_weights[:num_pos, :] = 1.0

        if pos_gt_depths is not None:
            depth_targets[:num_pos] = pos_gt_depths[:num_pos]
            depth_weights[:num_pos] = 1.0

        if pos_gt_alphas is not None:
            alpha_targets[:num_pos] = pos_gt_alphas[:num_pos]
            alpha_weights[:num_pos] = 1.0

        if pos_gt_rotys is not None:
            roty_targets[:num_pos] = pos_gt_rotys[:num_pos]
            roty_weights[:num_pos] = 1.0

        if pos_gt_dims is not None:
            dim_targets[:num_pos, :] = pos_gt_dims[:num_pos]
            dim_weights[:num_pos, :] = 1.0

        if pos_gt_2dc is not None:
            cen_2d_targets[:num_pos, :] = pos_gt_2dc[:num_pos]
            cen_2d_weights[:num_pos, :] = 1.0
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights, \
           depth_targets, depth_weights, alpha_targets, alpha_weights, \
           roty_targets, roty_weights, dim_targets, dim_weights, \
           cen_2d_targets, cen_2d_weights


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros(
        (bbox_targets.size(0), 4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros(
        (bbox_weights.size(0), 4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
