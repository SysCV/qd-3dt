import torch
import torch.nn as nn
import numpy as np
from addict import Dict
from pyquaternion import Quaternion

import mmcv
from qd3dt.apis import get_root_logger
from qd3dt.core import (bbox2roi, bbox2result, build_assigner, build_sampler,
                        track2results)
from .base import BaseDetector
from .test_3d_sep_uncertainty_mixins import RPNTestMixin, BBoxTestMixin
from .analyze_3d_mixins import Analyze3DMixin
from .. import builder
from ..registry import DETECTORS
from . import tracker
import scripts.tracking_utils as tu


@DETECTORS.register_module
class QuasiDense3DSepUncertainty(BaseDetector, RPNTestMixin, BBoxTestMixin,
                                 Analyze3DMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 bbox_3d_roi_extractor=None,
                 bbox_3d_head=None,
                 embed_roi_extractor=None,
                 embed_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(QuasiDense3DSepUncertainty, self).__init__()
        self.logger = get_root_logger()

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if bbox_3d_head is not None:
            self.bbox_3d_head = builder.build_head(bbox_3d_head)

        if embed_head is not None:
            self.embed_roi_extractor = builder.build_roi_extractor(
                embed_roi_extractor)
            self.embed_head = builder.build_head(embed_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.tracker = None

        self.init_weights(pretrained=pretrained)

        self.spool = len(self.bbox_roi_extractor.featmap_strides)
        self.espool = len(self.embed_roi_extractor.featmap_strides)

        self.counter = Dict(
            num_gt=torch.tensor([0]),
            num_fn=torch.tensor([0]),
            num_fp=torch.tensor([0]),
            num_idsw=torch.tensor([0]))
        self.writed = []

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_embed(self):
        return hasattr(self, 'embed_head') and self.embed_head is not None

    @property
    def with_bbox_3d(self):
        return hasattr(self, 'bbox_3d_head') and self.bbox_3d_head is not None

    def init_weights(self, pretrained=None):
        super(QuasiDense3DSepUncertainty, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_bbox_3d:
            self.bbox_3d_head.init_weights()
        if self.with_embed:
            self.embed_roi_extractor.init_weights()
            self.embed_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_pids=None,
                      gt_trans=None,
                      gt_alphas=None,
                      gt_rotys=None,
                      gt_dims=None,
                      gt_2dcs=None,
                      ref_img=None,
                      ref_img_meta=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      ref_gt_trans=None,
                      ref_gt_alphas=None,
                      ref_gt_rotys=None,
                      ref_gt_dims=None,
                      ref_gt_2dcs=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_pids=None):
        losses = dict()
        num_imgs = img.size(0)
        # extract backbone and neck features
        x = self.extract_feat(img)
        # region proposal network
        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(
            *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        # init proposal assigner and sampler
        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        # do proposal assign and sampling
        sampling_results = []
        for i in range(num_imgs):
            gt_pids_i = gt_pids[i] if gt_pids else None
            gt_depths_i = gt_trans[i][:, 2] if gt_trans else None
            gt_alphas_i = gt_alphas[i] if gt_alphas else None
            gt_rotys_i = gt_rotys[i] if gt_rotys else None
            gt_dims_i = gt_dims[i] if gt_dims else None
            gt_2dcs_i = gt_2dcs[i] if gt_2dcs else None
            ref_gt_bboxes_i = ref_gt_bboxes[i] if ref_gt_bboxes else None
            assign_result = bbox_assigner.assign(
                proposal_list[i],
                gt_bboxes[i],
                gt_bboxes_ignore=gt_bboxes_ignore[i],
                gt_labels=gt_labels[i],
                gt_depths=gt_depths_i,
                gt_alphas=gt_alphas_i,
                gt_rotys=gt_rotys_i,
                gt_dims=gt_dims_i,
                gt_2dcs=gt_2dcs_i,
                gt_pids=gt_pids_i)
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels=gt_labels[i],
                gt_depths=gt_depths_i,
                gt_alphas=gt_alphas_i,
                gt_rotys=gt_rotys_i,
                gt_dims=gt_dims_i,
                gt_2dcs=gt_2dcs_i,
                gt_pids=gt_pids_i,
                ref_gt_bboxes=ref_gt_bboxes_i,
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        # extract head features
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(x[:self.spool], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        # 3d estimation
        depth_pred, depth_uncertainty_pred, dim_pred, alpha_pred, cen_2d_pred = \
            self.bbox_3d_head(bbox_feats)
        bbox_3d_targets = self.bbox_3d_head.get_target(sampling_results,
                                                       gt_bboxes, gt_labels,
                                                       self.train_cfg.rcnn)
        loss_bbox_3d = self.bbox_3d_head.loss(cls_score, bbox_pred, depth_pred,
                                              depth_uncertainty_pred, dim_pred,
                                              alpha_pred, cen_2d_pred,
                                              *bbox_3d_targets)
        loss_bbox_3d_with_suffix = {
            key + '_3d': value
            for key, value in loss_bbox_3d.items()
        }
        losses.update(loss_bbox_3d_with_suffix)
        '''Embedding Part
        '''
        # init assigner and sampler
        embed_assigner = build_assigner(self.train_cfg.embed.assigner)
        embed_sampler = build_sampler(
            self.train_cfg.embed.sampler, context=self)
        key_sampling_results = []
        for i in range(num_imgs):
            # sample positive samples for key frame
            gt_pids_i = gt_pids[i] if gt_pids else None
            gt_depths_i = gt_trans[i][:, 2] if gt_trans else None
            gt_alphas_i = gt_alphas[i] if gt_alphas else None
            gt_rotys_i = gt_rotys[i] if gt_rotys else None
            gt_dims_i = gt_dims[i] if gt_dims else None
            gt_2dcs_i = gt_2dcs[i] if gt_2dcs else None
            ref_gt_bboxes_i = ref_gt_bboxes[i] if ref_gt_bboxes else None
            key_assign_result = embed_assigner.assign(
                proposal_list[i],
                gt_bboxes[i],
                gt_bboxes_ignore=gt_bboxes_ignore[i],
                gt_labels=gt_labels[i],
                gt_depths=gt_depths_i,
                gt_alphas=gt_alphas_i,
                gt_rotys=gt_rotys_i,
                gt_dims=gt_dims_i,
                gt_2dcs=gt_2dcs_i,
                gt_pids=gt_pids_i)
            key_sampling_result = embed_sampler.sample(
                key_assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels=gt_labels[i],
                gt_depths=gt_depths_i,
                gt_alphas=gt_alphas_i,
                gt_rotys=gt_rotys_i,
                gt_dims=gt_dims_i,
                gt_2dcs=gt_2dcs_i,
                gt_pids=gt_pids_i,
                ref_gt_bboxes=ref_gt_bboxes_i,
                feats=[lvl_feat[i][None] for lvl_feat in x])
            key_sampling_results.append(key_sampling_result)

        ref_x = self.extract_feat(ref_img)
        # get proposals for reference image
        if ref_gt_bboxes_ignore is None:
            ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        ref_rpn_outs = self.rpn_head(ref_x)
        ref_proposals_in = ref_rpn_outs + (ref_img_meta,
                                           self.train_cfg.rpn_proposal)
        ref_proposals = self.rpn_head.get_bboxes(*ref_proposals_in)
        ref_sampling_results = []
        for i in range(num_imgs):
            # sample pos / neg samples for reference frame
            ref_gt_pids_i = ref_gt_pids[i] if ref_gt_pids else None
            ref_gt_depths_i = ref_gt_trans[i][:, 2] if ref_gt_trans else None
            ref_gt_alphas_i = ref_gt_alphas[i] if ref_gt_alphas else None
            ref_gt_rotys_i = ref_gt_rotys[i] if ref_gt_rotys else None
            ref_gt_dims_i = ref_gt_dims[i] if ref_gt_dims else None
            ref_gt_2dcs_i = ref_gt_2dcs[i] if ref_gt_2dcs else None
            ref_gt_bboxes_i = ref_gt_bboxes[i] if ref_gt_bboxes else None
            ref_assign_result = embed_assigner.assign(
                ref_proposals[i],
                ref_gt_bboxes_i,
                gt_bboxes_ignore=ref_gt_bboxes_ignore[i],
                gt_labels=ref_gt_labels[i],
                gt_depths=ref_gt_depths_i,
                gt_alphas=ref_gt_alphas_i,
                gt_rotys=ref_gt_rotys_i,
                gt_dims=ref_gt_dims_i,
                gt_2dcs=ref_gt_2dcs_i,
                gt_pids=ref_gt_pids_i)
            ref_sampling_result = embed_sampler.sample(
                ref_assign_result,
                ref_proposals[i],
                ref_gt_bboxes_i,
                gt_labels=ref_gt_labels[i],
                gt_depths=ref_gt_depths_i,
                gt_alphas=ref_gt_alphas_i,
                gt_rotys=ref_gt_rotys_i,
                gt_dims=ref_gt_dims_i,
                gt_2dcs=ref_gt_2dcs_i,
                gt_pids=ref_gt_pids_i,
                ref_gt_bboxes=gt_bboxes[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)
        # select pairs for training
        # key pos embeds
        if self.train_cfg.embed.with_key_pos:
            key_pos_rois = bbox2roi(
                [res.pos_bboxes for res in key_sampling_results])
        else:
            key_pos_rois = bbox2roi(gt_bboxes)
        key_pos_feats = self.embed_roi_extractor(x[:self.espool], key_pos_rois)
        key_embeds, key_depth = self.embed_head(key_pos_feats)

        # ref gt embeds
        ref_gt_rois = bbox2roi(ref_gt_bboxes)
        ref_gt_feats = self.embed_roi_extractor(ref_x[:self.espool],
                                                ref_gt_rois)
        ref_gt_embeds, ref_gt_depth = self.embed_head(ref_gt_feats)

        if self.train_cfg.embed.with_ref_pos:
            ref_pos_rois = bbox2roi(
                [res.pos_bboxes for res in ref_sampling_results])
            ref_pos_feats = self.embed_roi_extractor(ref_x[:self.espool],
                                                     ref_pos_rois)
            ref_pos_embeds, ref_pos_depth = self.embed_head(ref_pos_feats)
        else:
            ref_pos_embeds, ref_pos_depth = None, None

        if self.train_cfg.embed.with_ref_neg:
            ref_neg_rois = bbox2roi(
                [res.neg_bboxes for res in ref_sampling_results])
            ref_neg_feats = self.embed_roi_extractor(ref_x[:self.espool],
                                                     ref_neg_rois)
            ref_neg_embeds, ref_neg_depth = self.embed_head(ref_neg_feats)
        else:
            ref_neg_embeds, ref_neg_depth = None, None

        if self.train_cfg.embed.with_key_neg:
            key_neg_rois = bbox2roi(
                [res.neg_bboxes for res in key_sampling_results])
            key_neg_feats = self.embed_roi_extractor(x[:self.espool],
                                                     key_neg_rois)
            key_neg_embeds, key_neg_depth = self.embed_head(key_neg_feats)
        else:
            key_neg_embeds, key_neg_depth = None, None

        # matching and cal loss
        matrix, cos_matrix = self.embed_head.match(
            key_embeds=key_embeds,
            ref_gt_embeds=ref_gt_embeds,
            ref_pos_embeds=ref_pos_embeds,
            ref_neg_embeds=ref_neg_embeds,
            key_neg_embeds=key_neg_embeds,
            key_sampling_results=key_sampling_results,
            ref_sampling_results=ref_sampling_results,
            img_meta=img_meta,
            cfg=self.train_cfg.embed)

        asso_targets = self.embed_head.get_asso_targets(
            key_sampling_results, gt_pids, self.train_cfg.embed)

        loss_embed = self.embed_head.cal_loss_embed(
            matrix,
            cos_matrix,
            *asso_targets,
            key_sampling_results=key_sampling_results,
            ref_sampling_results=ref_sampling_results,
            cfg=self.train_cfg.embed)
        losses.update(loss_embed)
        return losses

    def simple_test(self,
                    img,
                    img_meta,
                    pure_det=False,
                    proposals=None,
                    rescale=False):

        # init tracker
        frame_ind = img_meta[0].get('frame_id', -1)
        is_kitti = 'KITTI' in img_meta[0]['img_info']['file_name']
        use_3d_center = self.test_cfg.get('use_3d_center', False)

        if self.tracker is None:
            self.tracker = mmcv.runner.obj_from_dict(self.test_cfg.track,
                                                     tracker)
        elif img_meta[0].get('first_frame', False):
            num_tracklets = self.tracker.num_tracklets
            del self.tracker
            self.test_cfg.track.init_track_id = num_tracklets
            self.tracker = mmcv.runner.obj_from_dict(self.test_cfg.track,
                                                     tracker)

        mmcv.check_accum_time('detection', counting=True)
        x = self.extract_feat(img)
        # rpn
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) \
            if proposals is None \
            else proposals
        # bbox head
        det_bboxes, det_labels, det_depths, det_depths_uncertainty, det_dims, det_alphas, det_2dcs = \
            self.simple_test_bboxes(x, img_meta, proposal_list,
                                    self.test_cfg.rcnn, rescale=rescale)
        mmcv.check_accum_time('detection', counting=False)

        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        # return if only test for detection
        if frame_ind == -1:
            outputs = dict(
                bbox_results=bbox_results,
                depth_results=det_depths,
                depth_uncertainty_results=det_depths_uncertainty,
                dim_results=det_dims,
                alpha_results=det_alphas,
                cen_2ds_results=det_2dcs,
                track_results=track2results(
                    det_bboxes, det_labels,
                    det_labels.new_zeros(det_labels.size())))

            if self.test_cfg.get('save_txt', False):
                if img_meta[0]['img_info']['type'] == 'DET':
                    self.save_det_txt(
                        outputs,
                        self.test_cfg.save_txt,
                        img_meta,
                        use_3d_box_center=use_3d_center,
                        adjust_center=is_kitti)
                else:
                    self.save_trk_txt(
                        outputs,
                        self.test_cfg.save_txt,
                        img_meta,
                        use_3d_box_center=use_3d_center,
                        adjust_center=is_kitti)

            return outputs, use_3d_center

        mmcv.check_accum_time('embedding', counting=True)
        # re-pooling for embeddings
        if det_bboxes.size(0) != 0:
            bboxes = det_bboxes * img_meta[0]['scale_factor']
            embed_rois = bbox2roi([bboxes])
            embed_feats = self.embed_roi_extractor(x[:self.spool], embed_rois)
            embeds, emb_depth = self.embed_head(embed_feats)
        else:
            embeds = det_bboxes.new_zeros(
                [det_bboxes.shape[0], self.embed_head.embed_channels])
        mmcv.check_accum_time('embedding', counting=False)

        # save pkls for offline processing
        if self.test_cfg.get('save', False):
            self.save_pkl(
                img_meta,
                det_bboxes,
                det_labels,
                embeds,
                det_depths=det_depths,
                det_dims=det_dims,
                det_alphas=det_alphas,
                det_2dcs=det_2dcs)

        # TODO: use boxes_3d to match KF3d in tracker
        mmcv.check_accum_time('lifting', counting=True)
        projection = det_bboxes.new_tensor(img_meta[0]['calib'])
        position = det_bboxes.new_tensor(img_meta[0]['pose']['position'])
        r_camera_to_world = tu.angle2rot(
            np.array(img_meta[0]['pose']['rotation']))
        rotation = det_bboxes.new_tensor(r_camera_to_world)
        cam_rot_quat = Quaternion(matrix=r_camera_to_world)
        quat_det_yaws_world = {'roll_pitch': [], 'yaw_world': []}

        if det_depths is not None and det_2dcs is not None:
            corners = tu.imagetocamera_torch(det_2dcs, det_depths, projection)
            corners_global = tu.cameratoworld_torch(corners, position,
                                                    rotation)
            det_yaws = tu.alpha2yaw_torch(det_alphas, corners[:, 0:1],
                                          corners[:, 2:3])

            for det_yaw in det_yaws:
                yaw_quat = Quaternion(
                    axis=[0, 1, 0], radians=det_yaw.cpu().numpy())
                rotation_world = cam_rot_quat * yaw_quat
                if rotation_world.z < 0:
                    rotation_world *= -1
                roll_world, pitch_world, yaw_world = tu.quaternion_to_euler(
                    rotation_world.w, rotation_world.x, rotation_world.y,
                    rotation_world.z)
                quat_det_yaws_world['roll_pitch'].append(
                    [roll_world, pitch_world])
                quat_det_yaws_world['yaw_world'].append(yaw_world)

            det_yaws_world = rotation.new_tensor(
                np.array(quat_det_yaws_world['yaw_world'])[:, None])
            det_boxes_3d = torch.cat(
                [corners_global, det_yaws_world, det_dims], dim=1)
        else:
            det_boxes_3d = det_bboxes.new_zeros([det_bboxes.shape[0], 7])
        mmcv.check_accum_time('lifting', counting=False)

        mmcv.check_accum_time('tracking', counting=True)
        match_bboxes, match_labels, match_boxes_3ds, ids, inds, valids = \
            self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                boxes_3d=det_boxes_3d,
                depth_uncertainty=det_depths_uncertainty,
                position=position,
                rotation=rotation,
                embeds=embeds,
                cur_frame=frame_ind,
                pure_det=pure_det)
        mmcv.check_accum_time('tracking', counting=False)

        mmcv.check_accum_time('reproject', counting=True)
        if det_depths is not None and det_2dcs is not None:
            match_dims = match_boxes_3ds[:, -3:]
            match_corners_cam = tu.worldtocamera_torch(match_boxes_3ds[:, :3],
                                                       position, rotation)
            match_depths = match_corners_cam[:, 2:3]

            match_yaws = []
            for match_order, match_yaw in zip(
                    inds[valids].cpu().numpy(),
                    match_boxes_3ds[:, 3].cpu().numpy()):
                roll_world, pitch_world = quat_det_yaws_world['roll_pitch'][
                    match_order]
                rotation_cam = cam_rot_quat.inverse * Quaternion(
                    tu.euler_to_quaternion(roll_world, pitch_world, match_yaw))
                vtrans = np.dot(rotation_cam.rotation_matrix,
                                np.array([1, 0, 0]))
                match_yaws.append(-np.arctan2(vtrans[2], vtrans[0]))

            match_yaws = rotation.new_tensor(np.array(match_yaws)).unsqueeze(1)
            match_alphas = tu.yaw2alpha_torch(match_yaws,
                                              match_corners_cam[:, 0:1],
                                              match_corners_cam[:, 2:3])
            match_corners_frm = tu.cameratoimage_torch(match_corners_cam,
                                                       projection)
            match_2dcs = match_corners_frm
        else:
            if det_depths is not None:
                match_depths = det_depths[inds][valids]
            else:
                match_depths = None
            if det_2dcs is not None:
                match_2dcs = det_2dcs[inds][valids]
            else:
                match_2dcs = None
            if det_dims is not None:
                match_dims = det_dims[inds][valids]
            else:
                match_dims = None
            if det_alphas is not None:
                match_alphas = det_alphas[inds][valids]
            else:
                match_alphas = None
        mmcv.check_accum_time('reproject', counting=False)

        if self.test_cfg.get('analyze', False):
            self.analyze(
                img_meta=img_meta,
                bboxes=match_bboxes.cpu(),
                labels=match_labels.cpu(),
                depths=match_depths.cpu()
                if match_depths is not None else None,
                dims=match_dims.cpu() if match_dims is not None else None,
                alphas=match_alphas.cpu()
                if match_alphas is not None else None,
                cen_2ds=match_2dcs.cpu() if match_2dcs is not None else None,
                ids=ids.cpu(),
                save=True)

        # parse tracking results
        track_inds = ids > -1
        track_bboxes = match_bboxes[track_inds]
        track_labels = match_labels[track_inds]
        if match_depths is not None:
            track_depths = match_depths[track_inds]
        else:
            track_depths = None
        if match_dims is not None:
            track_dims = match_dims[track_inds]
        else:
            track_dims = None
        if match_alphas is not None:
            track_alphas = match_alphas[track_inds]
        else:
            track_alphas = None
        if match_2dcs is not None:
            track_2dcs = match_2dcs[track_inds]
        else:
            track_2dcs = None
        track_ids = ids[track_inds]
        track_results = track2results(track_bboxes, track_labels, track_ids)
        outputs = dict(
            bbox_results=bbox_results,
            depth_results=track_depths,
            depth_uncertainty_results=det_depths_uncertainty,
            dim_results=track_dims,
            alpha_results=track_alphas,
            cen_2ds_results=track_2dcs,
            track_results=track_results)
        # show or save_txt
        if self.test_cfg.get('show', False):
            self.plt_3d_tracklets(
                img_meta,
                track_bboxes.cpu().numpy(),
                track_labels.cpu().numpy(),
                track_depths.cpu().numpy()
                if match_depths is not None else None,
                track_dims.cpu().numpy() if match_dims is not None else None,
                track_alphas.cpu().numpy()
                if match_alphas is not None else None,
                track_2dcs.cpu().numpy() if match_2dcs is not None else None,
                track_ids.cpu().numpy())
        if self.test_cfg.get('save_txt', False):
            if img_meta[0]['img_info']['type'] == 'DET':
                self.save_det_txt(
                    outputs,
                    self.test_cfg.save_txt,
                    img_meta,
                    use_3d_box_center=use_3d_center,
                    adjust_center=is_kitti)
            else:
                self.save_trk_txt(
                    outputs,
                    self.test_cfg.save_txt,
                    img_meta,
                    use_3d_box_center=use_3d_center,
                    adjust_center=is_kitti)

        return outputs, use_3d_center
