import torch


class SamplingResult(object):

    def __init__(self,
                 pos_inds,
                 neg_inds,
                 bboxes,
                 gt_bboxes,
                 assign_result,
                 gt_flags,
                 ref_gt_bboxes=None):
        self.gt_bboxes = gt_bboxes
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

        if assign_result.gt_depths is not None:
            self.pos_gt_depths = assign_result.gt_depths[pos_inds]
        else:
            self.pos_gt_depths = None

        if assign_result.gt_alphas is not None:
            self.pos_gt_alphas = assign_result.gt_alphas[pos_inds]
        else:
            self.pos_gt_alphas = None

        if assign_result.gt_rotys is not None:
            self.pos_gt_rotys = assign_result.gt_rotys[pos_inds]
        else:
            self.pos_gt_rotys = None

        if assign_result.gt_dims is not None:
            self.pos_gt_dims = assign_result.gt_dims[pos_inds]
        else:
            self.pos_gt_dims = None

        if assign_result.gt_2dcs is not None:
            self.pos_gt_2dcs = assign_result.gt_2dcs[pos_inds]
        else:
            self.pos_gt_2dcs = None

        if assign_result.pids is not None:
            self.pos_gt_pids = assign_result.pids[pos_inds]
        else:
            self.pos_gt_pids = None

        if ref_gt_bboxes is not None:
            self.ref_gt_bboxes = ref_gt_bboxes
            self.pos_ref_gt_bboxes = ref_gt_bboxes[self.pos_gt_pids, :]
            no_match_ids = torch.nonzero(self.pos_gt_pids < 0)
            self.pos_ref_gt_bboxes[no_match_ids, :] = torch.zeros(1,
                                                                  4).cuda() - 1

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
