import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps,
                 labels=None,
                 gt_depths=None,
                 gt_alphas=None,
                 gt_rotys=None,
                 gt_dims=None,
                 gt_2dcs=None,
                 pids=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self.gt_depths = gt_depths
        self.gt_alphas = gt_alphas
        self.gt_rotys = gt_rotys
        self.gt_dims = gt_dims
        self.gt_2dcs = gt_2dcs
        self.pids = pids

    def add_gt_(self,
                gt_labels,
                gt_depths,
                gt_alphas,
                gt_rotys,
                gt_dims,
                gt_2dcs,
                gt_pids):
        self_inds = torch.arange(
                1, len(gt_labels) + 1, dtype=torch.long,
                device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
                [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
        if self.gt_depths is not None:
            self.gt_depths = torch.cat([gt_depths, self.gt_depths])
        if self.gt_alphas is not None:
            self.gt_alphas = torch.cat([gt_alphas, self.gt_alphas])
        if self.gt_rotys is not None:
            self.gt_rotys = torch.cat([gt_rotys, self.gt_rotys])
        if self.gt_dims is not None:
            self.gt_dims = torch.cat([gt_dims, self.gt_dims])
        if self.gt_2dcs is not None:
            self.gt_2dcs = torch.cat([gt_2dcs, self.gt_2dcs])
        if self.pids is not None:
            self.pids = torch.cat([gt_pids, self.pids])
