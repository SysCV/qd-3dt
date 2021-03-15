import torch
import numpy as np
from qd3dt.ops.nms import nms_wrapper
from qd3dt.ops.nms import batched_nms


def multiclass_nms_winds(multi_bboxes,
                         multi_scores,
                         score_thr,
                         nms_cfg,
                         max_num=-1,
                         score_factors=None):
    num_classes = multi_scores.shape[1]
    bboxes, labels, keep_inds = [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, cls_keep_inds = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        global_inds = torch.nonzero(cls_inds == 1).squeeze(1)
        keep_inds.append(global_inds[cls_keep_inds])
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        keep_inds = torch.cat(keep_inds)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            keep_inds = keep_inds[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        keep_inds = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels, keep_inds


# def multiclass_nms_winds(multi_bboxes,
#                          multi_scores,
#                          score_thr,
#                          nms_cfg,
#                          max_num=-1,
#                          score_factors=None):
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category
#     if multi_bboxes.shape[1] > 4:
#         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
#     else:
#         bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
#     scores = multi_scores[:, :-1]

#     # filter out boxes with low scores
#     valid_mask = scores > score_thr
#     bboxes = bboxes[valid_mask]
#     if score_factors is not None:
#         scores = scores * score_factors[:, None]
#     scores = scores[valid_mask]
#     labels = valid_mask.nonzero()[:, 1]

#     if bboxes.numel() == 0:
#         bboxes = multi_bboxes.new_zeros((0, 5))
#         labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
#         keep = multi_bboxes.new_zeros((0, ), dtype=torch.long)
#         return bboxes, labels, keep
#     dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
#     return dets[:max_num], labels[keep[:max_num]], keep[:max_num]


def bbox_jitter(bboxes, alpha, num):
    outs = []
    for bbox in bboxes:
        # xyxy2xwyh
        xwyh = torch.tensor([[(bbox[0] + bbox[2]) / 2, bbox[2] - bbox[0],
                              (bbox[1] + bbox[3]) / 2, bbox[3] - bbox[1]]],
                            dtype=torch.float)
        xwyh = xwyh.repeat(num, 1)
        w, h = xwyh[0, 1], xwyh[0, 3]
        offset_x, offset_y = w * alpha, h * alpha
        dx = torch.rand(num, 2) * 2 * offset_x - offset_x
        dy = torch.rand(num, 2) * 2 * offset_y - offset_y
        delta = torch.cat((dx, dy), dim=1)
        xwyh += delta
        jit_bboxes = torch.zeros_like(xwyh).to(bbox.device)
        jit_bboxes[:, 0] = xwyh[:, 0] - xwyh[:, 1] / 2
        jit_bboxes[:, 1] = xwyh[:, 2] - xwyh[:, 3] / 2
        jit_bboxes[:, 2] = xwyh[:, 0] + xwyh[:, 1] / 2
        jit_bboxes[:, 3] = xwyh[:, 2] + xwyh[:, 3] / 2
        jit_bboxes = torch.cat([bbox[None, :4], jit_bboxes], dim=0)
        outs.append(jit_bboxes)
    outs = torch.cat(outs, dim=0)
    return outs
