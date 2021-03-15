import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qd3dt.core import multi_apply, get_similarity, bbox_overlaps
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from ..utils import ConvModule
import torch.distributed as dist
import copy


@HEADS.register_module
class MultiPos3DTrackHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=1,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 embed_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_depth=None,
                 loss_asso=None,
                 loss_iou=None):
        super(MultiPos3DTrackHead, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels)
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)
        self.depth_embed = nn.Linear(embed_channels, 1)
        self.loss_asso_tau = loss_asso.pop('tau', -1)
        self.loss_asso = loss_asso
        self.loss_depth = loss_depth
        self.loss_iou = loss_iou

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def init_weights(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)
        nn.init.normal_(self.depth_embed.weight, 0, 0.01)
        nn.init.constant_(self.depth_embed.bias, 0)

    def forward(self, x):
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)
        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        x = self.fc_embed(x)
        depth = self.depth_embed(x)
        return x, depth

    def match(self,
              key_embeds=None,
              ref_gt_embeds=None,
              ref_pos_embeds=None,
              ref_neg_embeds=None,
              key_neg_embeds=None,
              key_sampling_results=None,
              ref_sampling_results=None,
              img_meta=None,
              cfg=None):
        matrix = []
        cos_matrix = []
        n = len(key_sampling_results)
        # get key embeds
        if cfg.with_key_pos:
            num_keys = [res.pos_bboxes.size(0) for res in key_sampling_results]
        else:
            num_keys = [res.num_gts for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_keys)

        # get ref gt embeds
        num_ref_gts = [
            res.ref_gt_bboxes.size(0) for res in key_sampling_results
        ]
        ref_gt_embeds = torch.split(ref_gt_embeds, num_ref_gts)

        if cfg.with_ref_pos:
            num_ref_pos = [
                res.pos_bboxes.size(0) for res in ref_sampling_results
            ]
            ref_pos_embeds = torch.split(ref_pos_embeds, num_ref_pos)

        # get ref neg embeds
        if cfg.with_ref_neg:
            num_ref_negs = [
                res.neg_bboxes.size(0) for res in ref_sampling_results
            ]
            ref_neg_embeds = torch.split(ref_neg_embeds, num_ref_negs)

        if cfg.with_key_neg:
            num_key_negs = [
                res.neg_bboxes.size(0) for res in key_sampling_results
            ]
            key_neg_embeds = torch.split(key_neg_embeds, num_key_negs)

        for i in range(n):
            ref_embeds = [ref_gt_embeds[i]]
            if cfg.with_ref_pos:
                ref_embeds.append(ref_pos_embeds[i])
            if cfg.with_ref_neg:
                ref_embeds.append(ref_neg_embeds[i])
            if cfg.with_key_neg:
                ref_embeds.append(key_neg_embeds[i])
            ref_embeds = torch.cat(ref_embeds, dim=0)
            _matrix = get_similarity(
                key_embeds[i], ref_embeds, tau=self.loss_asso_tau)
            matrix.append(_matrix)
            _cos_matrix = get_similarity(
                key_embeds[i], ref_embeds, norm=True, tau=-1)
            cos_matrix.append(_cos_matrix)

        return matrix, cos_matrix

    def get_asso_targets(self, sampling_results, gt_pids, cfg):
        ids = []
        id_weights = []
        for i, res in enumerate(sampling_results):
            if cfg.with_key_pos:
                _ids = res.pos_gt_pids
            else:
                _ids = gt_pids[i]
            _id_weights = torch.ones_like(_ids, dtype=torch.float)
            num_ref_gts = res.ref_gt_bboxes.size(0)
            gt_is_dummy = torch.nonzero(_ids == -1).squeeze()
            _ids[gt_is_dummy] = 0
            _id_weights[gt_is_dummy] = 0.
            ids.append(_ids)
            id_weights.append(_id_weights)
        return ids, id_weights

    def cal_loss_embed(self, asso_probs, cos_probs, ids, id_weights,
                       key_sampling_results, ref_sampling_results, cfg):
        losses = dict()
        batch_size = len(ids)
        loss_depth = []
        loss_asso = []
        loss_iou = []
        nelements = 0.

        # calculate per image loss
        for prob, cos_prob, cur_ids, cur_weights, res, key_res in zip(
                asso_probs, cos_probs, ids, id_weights, ref_sampling_results,
                key_sampling_results):
            valid_idx = torch.nonzero(cur_weights).squeeze()

            if len(valid_idx.size()) == 0:
                loss_asso.append(prob.new_zeros(1))
                if self.loss_depth is not None:
                    loss_depth.append(prob.new_zeros(1))
                if self.loss_iou is not None:
                    loss_iou.append(prob.new_zeros(1))
                continue

            num_ref = 0.
            num_ref += res.gt_bboxes.size(0)
            pids = [torch.arange(res.gt_bboxes.size(0)).long().to(prob.device)]
            if cfg.with_ref_pos:
                num_ref += res.pos_bboxes.size(0)
                ious = bbox_overlaps(res.pos_bboxes, res.gt_bboxes)
                pids.append(ious.max(dim=1)[1])
            if cfg.with_ref_neg:
                num_ref += res.neg_bboxes.size(0)
                pids.append((torch.ones(res.neg_bboxes.size(0)).long() *
                             -2).to(prob.device))
            if cfg.with_key_neg:
                num_ref += key_res.neg_bboxes.size(0)
                pids.append((torch.ones(key_res.neg_bboxes.size(0)).long() *
                             -2).to(prob.device))
            assert num_ref == prob.size(1)
            pids = torch.cat(pids, dim=0)
            pos_inds = (cur_ids.view(-1, 1) == pids.view(1, -1)).float()
            neg_inds = (cur_ids.view(-1, 1) != pids.view(1, -1)).float()
            exp_pos = (torch.exp(-1 * prob) * pos_inds).sum(dim=1)
            exp_neg = (torch.exp(prob.clamp(max=80)) * neg_inds).sum(dim=1)
            loss = torch.log(1 + exp_pos * exp_neg)
            loss_asso.append(
                ((loss * cur_weights).sum() / cur_weights.sum()).unsqueeze(0))

            if self.loss_depth is not None:
                pass

            if self.loss_iou is not None:
                dists = torch.abs(cos_prob - pos_inds)**2
                pos_points = torch.nonzero(pos_inds == 1)
                pos_dists = dists[pos_points[:, 0], pos_points[:, 1]]
                nelements += pos_dists.nelement()
                # neg
                neg_inds = torch.nonzero(pos_inds == 0)
                if self.loss_iou['sample_ratio'] > -1:
                    num_negs = pos_dists.nelement(
                    ) * self.loss_iou['sample_ratio']
                    if len(neg_inds) < num_negs:
                        num_negs = len(neg_inds)
                else:
                    num_negs = len(neg_inds)
                nelements += num_negs
                if self.loss_iou['hard_mining']:
                    _loss_neg = dists[neg_inds[:, 0],
                                      neg_inds[:, 1]].topk(num_negs)[0]
                else:
                    neg_inds = self.random_choice(neg_inds, num_negs)
                    _loss_neg = dists[neg_inds[:, 0], neg_inds[:, 1]]
                if self.loss_iou['margin'] > 0:
                    _loss_neg *= (_loss_neg > self.loss_iou['margin']).float()
                loss_iou.append(
                    (pos_dists.sum() + _loss_neg.sum()).unsqueeze(0))

        # average
        losses['loss_asso'] = torch.cat(
            loss_asso).sum() / batch_size * self.loss_asso['loss_weight']

        if self.loss_depth is not None:
            losses['loss_depth'] = torch.cat(
                loss_depth).sum() / batch_size * self.loss_depth['loss_weight']

        if self.loss_iou is not None:
            losses['loss_iou'] = (
                torch.cat(loss_iou).sum() /
                (nelements + 1e-6)) * self.loss_iou['loss_weight']

        return losses

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]
