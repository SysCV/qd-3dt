import torch

from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from ..geometry import bbox_overlaps


class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates

    def assign(self,
               bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               gt_depths=None,
               gt_alphas=None,
               gt_rotys=None,
               gt_dims=None,
               gt_2dcs=None,
               gt_pids=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
            gt_depths (Tensor, optional): Labels of k gt_depths, shape (k, ).
            gt_alphas (Tensor, optional): Labels of k gt_alphas, shape (k, ).
            gt_rotys (Tensor, optional): Labels of k gt_rotys, shape (k, ).
            gt_dims (Tensor, optional): Labels of k gt_dims, shape (k, 3).
            gt_2dcs (Tensor, optional): Lables of k 2d center, shape (k, 2).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :4]
        overlaps = bbox_overlaps(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(
            overlaps, 
            gt_labels=gt_labels,
            gt_depths=gt_depths,
            gt_alphas=gt_alphas,
            gt_rotys=gt_rotys,
            gt_dims=gt_dims,
            gt_2dcs=gt_2dcs,
            gt_pids=gt_pids)
        return assign_result

    def assign_wrt_overlaps(
        self,
        overlaps,
        gt_labels=None,
        gt_depths=None,
        gt_alphas=None,
        gt_rotys=None,
        gt_dims=None,
        gt_2dcs=None,
        gt_pids=None
        ):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).
            gt_depths (Tensor, optional): Labels of k gt_depths, shape (k, ).
            gt_alphas (Tensor, optional): Labels of k gt_alphas, shape (k, ).
            gt_rotys (Tensor, optional): Labels of k gt_rotys, shape (k, ).
            gt_dims (Tensor, optional): Labels of k gt_dims, shape (k, 3).
            gt_2dcs (Tensor, optional): Lables of k 2d center, shape (k, 2).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        if gt_depths is not None:
            assigned_depths = assigned_gt_inds.new_zeros(
                                    (num_bboxes, ), 
                                    dtype=torch.float)
            if pos_inds.numel() > 0:
                assigned_depths[pos_inds] = gt_depths[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_depths = None

        if gt_alphas is not None:
            assigned_alphas = assigned_gt_inds.new_zeros(
                    (num_bboxes, ),
                    dtype=torch.float)
            if pos_inds.numel() > 0:
                assigned_alphas[pos_inds] = gt_alphas[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_alphas = None

        if gt_rotys is not None:
            assigned_rotys = assigned_gt_inds.new_zeros(
                    (num_bboxes, ),
                    dtype=torch.float)
            if pos_inds.numel() > 0:
                assigned_rotys[pos_inds] = gt_rotys[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_rotys = None

        if gt_dims is not None:
            assigned_dims = assigned_gt_inds.new_zeros(
                    (num_bboxes, 3),
                    dtype=torch.float)
            if pos_inds.numel() > 0:
                assigned_dims[pos_inds] = gt_dims[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_dims = None

        if gt_2dcs is not None:
            assigned_2dcs = assigned_gt_inds.new_zeros(
                    (num_bboxes, 2),
                    dtype=torch.float)
            if pos_inds.numel() > 0:
                assigned_2dcs[pos_inds] = gt_2dcs[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_2dcs = None

        if gt_pids is not None:
            assigned_pids = assigned_gt_inds.new_zeros((num_bboxes, ))
            if pos_inds.numel() > 0:
                assigned_pids[pos_inds] = gt_pids[assigned_gt_inds[pos_inds] -
                                                  1]
        else:
            assigned_pids = None

        return AssignResult(
            num_gts,
            assigned_gt_inds,
            max_overlaps,
            labels=assigned_labels,
            gt_depths=assigned_depths,
            gt_alphas=assigned_alphas,
            gt_rotys=assigned_rotys,
            gt_dims=assigned_dims,
            gt_2dcs=assigned_2dcs,
            pids=assigned_pids)
