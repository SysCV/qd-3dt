import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qd3dt.core import delta2bbox, force_fp32
from qd3dt.ops.nms import batched_nms
from qd3dt.core.bbox.bbox_3d_target_2dcs import bbox_3d_target

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from ..losses import accuracy


@HEADS.register_module
class ConvFCBBox3DRotSepConfidenceHead(nn.Module):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_dep_convs=0,
                 num_dep_fcs=0,
                 num_dim_convs=0,
                 num_dim_fcs=0,
                 num_rot_convs=0,
                 num_rot_fcs=0,
                 num_2dc_convs=0,
                 num_2dc_fcs=0,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=8,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 center_scale=10,
                 reg_class_agnostic=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_depth=None,
                 with_depth=False,
                 loss_uncertainty=None,
                 with_uncertainty=False,
                 use_uncertainty=False,
                 loss_dim=None,
                 with_dim=False,
                 loss_rot=None,
                 with_rot=False,
                 loss_2dc=None,
                 with_2dc=False,
                 loss_cls=None,
                 loss_bbox=None,
                 *args,
                 **kwargs):
        super(ConvFCBBox3DRotSepConfidenceHead, self).__init__()
        assert (num_shared_convs + num_shared_fcs + num_dep_convs +
                num_dep_fcs + num_dim_convs + num_dim_fcs + num_rot_convs +
                num_rot_fcs + num_2dc_convs + num_2dc_fcs > 0)
        if num_dep_convs > 0 or num_dim_convs > 0 \
                or num_rot_convs or num_2dc_convs:
            assert num_shared_fcs == 0
        if not with_depth:
            assert num_dep_convs == 0 and num_dep_fcs == 0
        if not with_dim:
            assert num_dim_convs == 0 and num_dim_fcs == 0
        if not with_rot:
            assert num_rot_convs == 0 and num_rot_fcs == 0
        if not with_2dc:
            assert num_2dc_convs == 0 and num_2dc_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_dep_convs = num_dep_convs
        self.num_dep_fcs = num_dep_fcs
        self.num_dim_convs = num_dim_convs
        self.num_dim_fcs = num_dim_fcs
        self.num_rot_convs = num_rot_convs
        self.num_rot_fcs = num_rot_fcs
        self.num_2dc_convs = num_2dc_convs
        self.num_2dc_fcs = num_2dc_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.reg_class_agnostic = reg_class_agnostic
        self.target_means = target_means
        self.target_stds = target_stds
        self.center_scale = center_scale
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_depth = build_loss(loss_depth) if with_depth else None
        self.with_depth = with_depth
        self.loss_uncertainty = build_loss(
            loss_uncertainty) if with_uncertainty else None
        self.with_uncertainty = with_uncertainty
        self.use_uncertainty = use_uncertainty
        self.loss_dim = build_loss(loss_dim) if with_dim else None
        self.with_dim = with_dim
        self.loss_rot = build_loss(loss_rot) if with_rot else None
        self.with_rot = with_rot
        self.loss_2dc = build_loss(loss_2dc) if with_2dc else None
        self.with_2dc = with_2dc
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.cls_out_channels = num_classes
        self.num_classes = num_classes

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add depth specific branch
        self.dep_convs, self.dep_fcs, self.dep_last_dim = \
            self._add_conv_fc_branch(
                self.num_dep_convs, self.num_dep_fcs, self.shared_out_channels)

        # add dim specific branch
        self.dim_convs, self.dim_fcs, self.dim_last_dim = \
            self._add_conv_fc_branch(
                self.num_dim_convs, self.num_dim_fcs, self.shared_out_channels)

        # add rot specific branch
        self.rot_convs, self.rot_fcs, self.rot_last_dim = \
            self._add_conv_fc_branch(
                self.num_rot_convs, self.num_rot_fcs,
                self.shared_out_channels)

        # add 2dc specific branch
        self.cen_2d_convs, self.cen_2d_fcs, self.cen_2d_last_dim = \
            self._add_conv_fc_branch(
                self.num_2dc_convs, self.num_2dc_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0:
            if self.num_dep_fcs == 0:
                self.dep_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_dim_fcs == 0:
                self.dim_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_rot_fcs == 0:
                self.rot_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_2dc_fcs == 0:
                self.cen_2d_last_dim *= (
                    self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_depth:
            out_dim_dep = (1 if self.reg_class_agnostic else
                           self.cls_out_channels)
            if self.with_uncertainty:
                self.fc_dep_uncer = nn.Linear(self.dep_last_dim, out_dim_dep)
            self.fc_dep = nn.Linear(self.dep_last_dim, out_dim_dep)
        if self.with_dim:
            out_dim_size = (3 if self.reg_class_agnostic else 3 *
                            self.cls_out_channels)
            self.fc_dim = nn.Linear(self.dim_last_dim, out_dim_size)
        if self.with_rot:
            out_rot_size = (8 if self.reg_class_agnostic else 8 *
                            self.cls_out_channels)
            self.fc_rot = nn.Linear(self.rot_last_dim, out_rot_size)
        if self.with_2dc:
            out_2dc_size = (2 if self.reg_class_agnostic else 2 *
                            self.cls_out_channels)
            self.fc_2dc = nn.Linear(self.cen_2d_last_dim, out_2dc_size)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0):
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        module_lists = [self.shared_fcs]
        if self.with_depth:
            if self.with_uncertainty:
                module_lists += [self.fc_dep_uncer]
            module_lists += [self.fc_dep, self.dep_fcs]
        if self.with_dim:
            module_lists += [self.fc_dim, self.dim_fcs]
        if self.with_rot:
            module_lists += [self.fc_rot, self.rot_fcs]
        if self.with_2dc:
            module_lists += [self.fc_2dc, self.cen_2d_fcs]

        for module_list in module_lists:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def get_embeds(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_dep = x
        x_dim = x
        x_rot = x
        x_2dc = x

        for conv in self.dep_convs:
            x_dep = conv(x_dep)
        if x_dep.dim() > 2:
            x_dep = x_dep.view(x_dep.size(0), -1)
        for fc in self.dep_fcs:
            x_dep = self.relu(fc(x_dep))

        for conv in self.dim_convs:
            x_dim = conv(x_dim)
        if x_dim.dim() > 2:
            x_dim = x_dim.view(x_dim.size(0), -1)
        for fc in self.dim_fcs:
            x_dim = self.relu(fc(x_dim))

        for conv in self.rot_convs:
            x_rot = conv(x_rot)
        if x_rot.dim() > 2:
            x_rot = x_rot.view(x_rot.size(0), -1)
        for fc in self.rot_fcs:
            x_rot = self.relu(fc(x_rot))

        for conv in self.cen_2d_convs:
            x_2dc = conv(x_2dc)
        if x_2dc.dim() > 2:
            x_2dc = x_2dc.view(x_2dc.size(0), -1)
        for fc in self.cen_2d_fcs:
            x_2dc = self.relu(fc(x_2dc))

        return x_dep, x_dim, x_rot, x_2dc

    def get_logits(self, x_dep, x_dim, x_rot, x_2dc):

        def get_rot(pred):
            pred = pred.view(pred.size(0), -1, 8)

            # bin 1
            divider1 = torch.sqrt(pred[:, :, 2:3]**2 + pred[:, :, 3:4]**2 +
                                  1e-10)
            b1sin = pred[:, :, 2:3] / divider1
            b1cos = pred[:, :, 3:4] / divider1

            # bin 2
            divider2 = torch.sqrt(pred[:, :, 6:7]**2 + pred[:, :, 7:8]**2 +
                                  1e-10)
            b2sin = pred[:, :, 6:7] / divider2
            b2cos = pred[:, :, 7:8] / divider2

            rot = torch.cat(
                [pred[:, :, 0:2], b1sin, b1cos, pred[:, :, 4:6], b2sin, b2cos],
                2)
            return rot

        depth_pred = self.fc_dep(x_dep) if self.with_depth else None
        depth_uncertainty_pred = self.fc_dep_uncer(
            x_dep) if self.with_depth and self.with_uncertainty else None
        dim_pred = self.fc_dim(x_dim) if self.with_dim else None
        rot_pred = get_rot(self.fc_rot(x_rot)) if self.with_rot else None
        cen_2d_pred = self.fc_2dc(x_2dc) if self.with_2dc else None

        return depth_pred, depth_uncertainty_pred, dim_pred, rot_pred, cen_2d_pred

    def forward(self, x):
        x_dep, x_dim, x_rot, x_2dc = self.get_embeds(x)
        depth_pred, depth_uncertainty_pred, dim_pred, rot_pred, cen_2d_pred = \
            self.get_logits(x_dep, x_dim, x_rot, x_2dc)

        return depth_pred, depth_uncertainty_pred, dim_pred, rot_pred, cen_2d_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        pos_gt_depths = [res.pos_gt_depths for res in sampling_results]
        pos_gt_alphas = [res.pos_gt_alphas for res in sampling_results]
        pos_gt_rotys = [res.pos_gt_rotys for res in sampling_results]
        pos_gt_dims = [res.pos_gt_dims for res in sampling_results]
        pos_gt_2dcs = [res.pos_gt_2dcs for res in sampling_results]

        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_3d_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            pos_gt_depths,
            pos_gt_alphas,
            pos_gt_rotys,
            pos_gt_dims,
            pos_gt_2dcs,
            rcnn_train_cfg,
            reg_classes=reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    @force_fp32(
        apply_to=('cls_score', 'bbox_pred', 'depth_pred',
                  'depth_uncertainty_pred', 'dim_pred', 'rot_pred',
                  'cen_2d_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             depth_pred,
             depth_uncertainty_pred,
             dim_pred,
             rot_pred,
             cen_2d_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             depth_targets,
             depth_weights,
             alpha_targets,
             alpha_weights,
             roty_targets,
             roty_weights,
             dim_targets,
             dim_weights,
             cen_2d_targets,
             cen_2d_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)

        pos_inds = labels > 0
        if bbox_pred is not None:
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)

        if depth_pred is not None and self.with_depth:
            depth_weights[depth_targets <= 0] = 0

            def get_depth_gt(gt, scale: float = 2.0):
                return torch.where(gt > 0,
                                   torch.log(gt) * scale, -gt.new_ones(1))

            if self.reg_class_agnostic:
                pos_depth_pred = depth_pred[pos_inds].flatten()
            else:
                pos_depth_pred = depth_pred.view(
                    bbox_pred.size(0), -1, 1)[pos_inds,
                                              labels[pos_inds]].flatten()

            pos_depth_targets = get_depth_gt(depth_targets[pos_inds])
            pos_depth_weights = depth_weights[pos_inds]
            losses['loss_depth'] = self.loss_depth(
                pos_depth_pred, pos_depth_targets, weight=pos_depth_weights)

            if depth_uncertainty_pred is not None and self.with_uncertainty:
                pos_depth_self_labels = torch.exp(
                    -torch.abs(pos_depth_pred - pos_depth_targets) * 5.0)

                pos_depth_self_weights = torch.where(
                    pos_depth_self_labels > 0.8,
                    pos_depth_weights.new_ones(1) * 5.0,
                    pos_depth_weights.new_ones(1) * 0.1)

                if self.reg_class_agnostic:
                    pos_depth_uncertainty_pred = depth_uncertainty_pred[
                        pos_inds].flatten()
                else:
                    pos_depth_uncertainty_pred = depth_uncertainty_pred.view(
                        bbox_pred.size(0), -1, 1)[pos_inds,
                                                  labels[pos_inds]].flatten()

                losses['loss_unc'] = self.loss_uncertainty(
                    pos_depth_uncertainty_pred,
                    pos_depth_self_labels.detach().clone(),
                    pos_depth_self_weights,
                    reduction_override=reduction_override)
                losses['unc_acc'] = accuracy(
                    torch.cat([
                        1.0 - pos_depth_uncertainty_pred[:, None],
                        pos_depth_uncertainty_pred[:, None]
                    ],
                              dim=1),
                    (pos_depth_self_labels > 0.8).detach().clone())

        if dim_pred is not None and self.with_dim:
            dim_weights[dim_targets <= 0] = 0

            def get_dim_gt(gt, scale: float = 2.0):
                return torch.where(gt > 0,
                                   torch.log(gt) * scale, gt.new_ones(1))

            if self.reg_class_agnostic:
                pos_dim_pred = dim_pred[pos_inds]
            else:
                pos_dim_pred = dim_pred.view(bbox_pred.size(0), -1,
                                             3)[pos_inds, labels[pos_inds]]
            pos_dim_targets = get_dim_gt(dim_targets[pos_inds])
            pos_dim_weights = dim_weights[pos_inds]
            losses['loss_dim'] = self.loss_dim(
                pos_dim_pred, pos_dim_targets, weight=pos_dim_weights)

        if rot_pred is not None and self.with_rot:
            alpha_weights[alpha_targets <= -10] = 0

            def get_rot_bin_gt(alpha_targets):
                bin_cls = alpha_targets.new_zeros(
                    (len(alpha_targets), 2)).long()
                bin_res = alpha_targets.new_zeros(
                    (len(alpha_targets), 2)).float()

                for i in range(len(alpha_targets)):
                    if alpha_targets[i] < np.pi / 6. or alpha_targets[
                            i] > 5 * np.pi / 6.:
                        bin_cls[i, 0] = 1
                        bin_res[i, 0] = alpha_targets[i] - (-0.5 * np.pi)

                    if alpha_targets[i] > -np.pi / 6. or alpha_targets[
                            i] < -5 * np.pi / 6.:
                        bin_cls[i, 1] = 1
                        bin_res[i, 1] = alpha_targets[i] - (0.5 * np.pi)
                return bin_cls, bin_res

            if self.reg_class_agnostic:
                pos_rot_pred = rot_pred[pos_inds].squeeze(1)
            else:
                pos_rot_pred = rot_pred.view(bbox_pred.size(0), -1,
                                             8)[pos_inds, labels[pos_inds]]
            pos_rot_cls, pos_rot_res = get_rot_bin_gt(alpha_targets[pos_inds])
            pos_rot_weights = alpha_weights[pos_inds]
            avg_factor = max(
                torch.sum(pos_rot_weights > 0).float().item(), 1.0)
            losses['loss_rot'] = self.loss_rot(
                pos_rot_pred,
                pos_rot_cls,
                pos_rot_res,
                weight=pos_rot_weights,
                avg_factor=avg_factor)

        if cen_2d_pred is not None and self.with_2dc:
            pos_2dc_weights = cen_2d_weights[pos_inds]

            def get_2dc_gt(gt_cen, scale: float = 10.0):
                return gt_cen / scale

            if self.reg_class_agnostic:
                pos_2dc_pred = cen_2d_pred[pos_inds]
            else:
                pos_2dc_pred = cen_2d_pred.view(bbox_pred.size(0), -1,
                                                2)[pos_inds, labels[pos_inds]]
            pos_2dc_targets = get_2dc_gt(
                cen_2d_targets[pos_inds], scale=self.center_scale)
            losses['loss_2dc'] = self.loss_2dc(
                pos_2dc_pred, pos_2dc_targets, weight=pos_2dc_weights)

        return losses

    @force_fp32(
        apply_to=('cls_score', 'bbox_pred', 'depth_pred', 'dim_pred',
                  'rot_pred', 'cen_2d_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       depth_pred,
                       depth_uncertainty_pred,
                       dim_pred,
                       rot_pred,
                       cen_2d_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            bboxes /= scale_factor

        def get_depth(pred, scale: float = 2.0):
            return torch.exp(pred / scale).view(pred.size(0), -1, 1)

        def get_uncertainty_prob(depth_uncertainty_pred):
            if depth_uncertainty_pred is None:
                return None
            return torch.clamp(
                depth_uncertainty_pred, min=0.0, max=1.0)

        def get_dim(pred, scale: float = 2.0):
            return torch.exp(pred / scale).view(pred.size(0), -1, 3)

        def get_alpha(rot):
            """Generate alpha value from predicted CLSREG array

            Args:
                rot (torch.Tensor): rotation CLSREG array: (B, num_classes, 8) 
                [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]

            Returns:
                torch.Tensor: (B, num_classes, 1)
            """
            alpha1 = torch.atan(
                rot[:, :, 2:3] / rot[:, :, 3:4]) + (-0.5 * np.pi)
            alpha2 = torch.atan(
                rot[:, :, 6:7] / rot[:, :, 7:8]) + (0.5 * np.pi)
            # Model is not decisive at index overlap region
            # Could be unstable for the alpha estimation
            # idx1 = (rot[:, :, 1:2] > rot[:, :, 0:1]).float()
            # idx2 = (rot[:, :, 5:6] > rot[:, :, 4:5]).float()
            # alpha = (alpha1 * idx1 + alpha2 * idx2)
            # alpha /= (idx1 + idx2 + ((idx1 + idx2) == 0))
            idx = (rot[:, :, 1:2] > rot[:, :, 5:6]).float()
            alpha = alpha1 * idx + alpha2 * (1 - idx)
            return alpha

        def get_delta_2d(delta_cen, scale: float = 10.0):
            return delta_cen.view(delta_cen.size(0), -1, 2) * scale

        def get_box_cen(bbox):
            return torch.cat(
                [(bbox[:, 0::4, None] + bbox[:, 2::4, None]) / 2.0,
                 (bbox[:, 1::4, None] + bbox[:, 3::4, None]) / 2.0],
                dim=2)

        def get_cen2d(delta_2d, box_cen):
            return delta_2d + box_cen

        depth_pred = get_depth(depth_pred)
        depth_uncertainty_prob = get_uncertainty_prob(depth_uncertainty_pred)
        if not self.use_uncertainty:
            depth_uncertainty_prob = depth_uncertainty_prob * 0. + 1.0
        dim_pred = get_dim(dim_pred)
        rot_pred = get_alpha(rot_pred)
        delta_2d = get_delta_2d(cen_2d_pred, scale=self.center_scale)
        cen2d_pred = get_cen2d(delta_2d, get_box_cen(bboxes.detach()))

        if cfg is None:
            return bboxes, scores, depth_pred, depth_uncertainty_prob, dim_pred, rot_pred, cen2d_pred
        else:
            det_bboxes, det_labels, det_depths, det_depth_uncertainty, dim_preds, rot_preds, cen2d_preds = \
                multiclass_3d_nms(
                    bboxes,
                    scores,
                    depth_pred,
                    depth_uncertainty_prob,
                    dim_pred,
                    rot_pred,
                    cen2d_pred,
                    cfg.score_thr, cfg.nms,
                    cfg.max_per_img)
            return det_bboxes, det_labels, det_depths, det_depth_uncertainty, dim_preds, rot_preds, cen2d_preds


def multiclass_3d_nms(multi_bboxes,
                      multi_scores,
                      depth_pred,
                      depth_uncertainty_pred,
                      dim_pred,
                      rot_pred,
                      cen_2d_pred,
                      score_thr,
                      nms_cfg,
                      max_num=-1,
                      score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        depth_pred (Tensor): shape (n, 1), estimated depth.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        reg_class_agnostic = False
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        reg_class_agnostic = True
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    if depth_uncertainty_pred is not None:
        valid_mask = (scores * depth_uncertainty_pred[:, 1:]) > score_thr
    else:
        valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]

    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        depths = multi_bboxes.new_zeros((0, ))
        depths_uncertainty = multi_bboxes.new_zeros((0, ))
        dims = multi_bboxes.new_zeros((0, 3))
        rots = multi_bboxes.new_zeros((0, ))
        cen_2ds = multi_bboxes.new_zeros((0, 2))
        return bboxes, labels, depths, depths_uncertainty, dims, rots, cen_2ds

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if depth_pred is not None:
        if reg_class_agnostic:
            depths = depth_pred.expand(-1, num_classes, -1)[valid_mask]
        else:
            depths = depth_pred.view(multi_scores.size(0), -1,
                                     1)[:, 1:][valid_mask]
        depths = depths[keep[:max_num]]
    else:
        depths = None

    if depth_uncertainty_pred is not None:
        if reg_class_agnostic:
            depths_uncertainty = depth_uncertainty_pred.expand(
                -1, num_classes, -1)[valid_mask]
        else:
            depths_uncertainty = depth_uncertainty_pred.view(
                multi_scores.size(0), -1, 1)[:, 1:][valid_mask]
        depths_uncertainty = depths_uncertainty[keep[:max_num]]
    else:
        depths_uncertainty = None

    if dim_pred is not None:
        if reg_class_agnostic:
            dims = dim_pred.expand(-1, num_classes, -1)[valid_mask]
        else:
            dims = dim_pred.view(multi_scores.size(0), -1, 3)[:,
                                                              1:][valid_mask]
        dims = dims[keep[:max_num]]
    else:
        dims = None

    if rot_pred is not None:
        if reg_class_agnostic:
            rots = rot_pred.expand(-1, num_classes, -1)[valid_mask]
        else:
            rots = rot_pred.view(multi_scores.size(0), -1, 1)[:,
                                                              1:][valid_mask]
        rots = rots[keep[:max_num]]
    else:
        rots = None

    if cen_2d_pred is not None:
        if reg_class_agnostic:
            cen_2ds = cen_2d_pred.expand(-1, num_classes, -1)[valid_mask]
        else:
            cen_2ds = cen_2d_pred.view(multi_scores.size(0), -1,
                                       2)[:, 1:][valid_mask]
        cen_2ds = cen_2ds[keep[:max_num]]
    else:
        cen_2ds = None

    return dets[:max_num], labels[keep[:max_num]], \
           depths, depths_uncertainty, dims, rots, cen_2ds


@HEADS.register_module
class SharedFCBBox3DRotSepConfidenceHead(ConvFCBBox3DRotSepConfidenceHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBox3DRotSepConfidenceHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_dep_convs=0,
            num_dep_fcs=0,
            num_dim_convs=0,
            num_dim_fcs=0,
            num_rot_convs=0,
            num_rot_fcs=0,
            num_2dc_convs=0,
            num_2dc_fcs=0,
            roi_feat_size=7,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            num_classes=8,
            reg_class_agnostic=True,
            conv_cfg=None,
            norm_cfg=None,
            loss_depth=None,
            with_depth=False,
            loss_dim=None,
            with_dim=False,
            loss_rot=None,
            with_rot=False,
            loss_2dc=None,
            with_2dc=False,
            loss_cls=None,
            loss_bbox=None,
            *args,
            **kwargs)
