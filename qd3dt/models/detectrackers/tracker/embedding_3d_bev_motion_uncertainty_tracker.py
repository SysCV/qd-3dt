import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

import platform
import mmcv
from .motion_model import get_lstm_model
from .tracker_model import get_tracker
import scripts.tracking_utils as tu

if platform.system() == 'Linux':
    from qd3dt.core import bbox_overlaps
    mother = object
else:
    from qd3dt.core.bbox.geometry import bbox_overlaps
    from qd3dt.models.detectrackers.analyze_3d_mixins import Analyze3DMixin
    mother = Analyze3DMixin


class Embedding3DBEVMotionUncertaintyTracker(mother):

    def __init__(
            self,
            init_score_thr: float = 0.8,
            init_track_id: int = 0,
            obj_score_thr: float = 0.5,
            match_score_thr: float = 0.5,
            memo_tracklet_frames: int = 10,
            memo_backdrop_frames: int = 1,
            memo_momentum: float = 0.8,
            motion_momentum: float = 0.8,
            nms_conf_thr: float = 0.5,
            nms_backdrop_iou_thr: float = 0.3,
            nms_class_iou_thr: float = 0.7,
            loc_dim: int = 7,
            with_deep_feat: bool = True,
            with_cats: bool = True,
            with_bbox_iou: bool = True,
            with_depth_ordering: bool = True,
            with_depth_uncertainty: bool = True,
            tracker_model_name: str = 'KalmanBox3DTracker',
            lstm_name: str = 'LSTM',
            lstm_ckpt_name: str = './checkpoints/803_kitti_300_linear.pth',
            track_bbox_iou: str = 'bbox',
            depth_match_metric: str = 'centriod',
            match_metric: str = 'cycle_softmax',
            match_algo: str = 'greedy'):
        assert 0 <= memo_momentum <= 1.0
        assert 0 <= motion_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.motion_momentum = motion_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_deep_feat = with_deep_feat
        self.with_cats = with_cats
        self.with_depth_ordering = with_depth_ordering
        self.with_depth_uncertainty = with_depth_uncertainty
        self.with_bbox_iou = with_bbox_iou
        assert track_bbox_iou in ['box2d', 'box2d_depth_aware', 'bev', 'box3d']
        self.track_bbox_iou = track_bbox_iou
        assert depth_match_metric in [
            'centroid', 'cosine', 'pure_motion', 'motion'
        ]
        self.depth_match_metric = depth_match_metric
        self.tracker_model = get_tracker(tracker_model_name)
        self.tracker_model_name = tracker_model_name
        self.loc_dim = loc_dim
        if tracker_model_name == 'LSTM3DTracker':
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.lstm = get_lstm_model(lstm_name)(1, 64, 128, 2,
                                                  self.loc_dim).to(self.device)
            self.lstm.eval()
            ckpt = torch.load(lstm_ckpt_name)
            print(f"\nLoad {lstm_name} checkpoints {lstm_ckpt_name} "
                  f"with {ckpt['epoch']} epochs")
            try:
                self.lstm.load_state_dict(ckpt['state_dict'])
            except (RuntimeError, KeyError) as ke:
                print("Cannot load full model: {}".format(ke))
                state = self.lstm.state_dict()
                # ckpt['state_dict']["pred2vel.weight"] = ckpt['state_dict']["pred2loc.weight"]
                # del ckpt['state_dict']["pred2loc.weight"]
                state.update(ckpt['state_dict'])
                self.lstm.load_state_dict(state)
            print(f"=> Successfully loaded checkpoint {lstm_ckpt_name}")
            del ckpt
        assert match_metric in ['cycle_softmax', 'softmax', 'cosine']
        self.match_metric = match_metric
        self.match_algo = match_algo

        self._debug = False
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        self.bbox_affinity_weight = 0.5 if with_deep_feat else 1.0
        self.feat_affinity_weight = 1 - self.bbox_affinity_weight if with_bbox_iou else 1.0

        self.num_tracklets = init_track_id
        self.tracklets = dict()
        self.backdrops = []

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, boxes_3d, depth_uncertainty, embeds,
                    labels, cur_frame):
        tracklet_inds = ids > -1

        # update memo
        for tid, bbox, box_3d, d_uncertainty, embed, label in zip(
                ids[tracklet_inds], bboxes[tracklet_inds],
                boxes_3d[tracklet_inds], depth_uncertainty[tracklet_inds],
                embeds[tracklet_inds], labels[tracklet_inds]):
            tid = int(tid)
            if tid in self.tracklets.keys():
                self.tracklets[tid]['bbox'] = bbox

                mmcv.check_accum_time('update', counting=True)
                self.tracklets[tid]['tracker'].update(
                    box_3d.cpu().numpy(),
                    d_uncertainty.cpu().numpy())
                mmcv.check_accum_time('update', counting=False)

                tracker_box = self.tracklets[tid]['tracker'].get_state()[:7]
                pd_box_3d = box_3d.new_tensor(tracker_box)

                velocity = (pd_box_3d - self.tracklets[tid]['box_3d']) / (
                    cur_frame - self.tracklets[tid]['last_frame'])

                self.tracklets[tid]['box_3d'] = pd_box_3d
                self.tracklets[tid]['embed'] += self.memo_momentum * (
                    embed - self.tracklets[tid]['embed'])
                self.tracklets[tid]['label'] = label
                self.tracklets[tid]['velocity'] = (
                    self.tracklets[tid]['velocity'] *
                    self.tracklets[tid]['acc_frame'] + velocity) / (
                        self.tracklets[tid]['acc_frame'] + 1)
                self.tracklets[tid]['last_frame'] = cur_frame
                self.tracklets[tid]['acc_frame'] += 1
            else:
                built_tracker = self.tracker_model(
                    self.device, self.lstm,
                    box_3d.cpu().numpy(),
                    d_uncertainty.cpu().numpy()
                ) if self.tracker_model_name == 'LSTM3DTracker' else self.tracker_model(
                    box_3d.cpu().numpy(),
                    d_uncertainty.cpu().numpy())
                self.tracklets[tid] = dict(
                    bbox=bbox,
                    box_3d=box_3d,
                    tracker=built_tracker,
                    embed=embed,
                    label=label,
                    last_frame=cur_frame,
                    velocity=torch.zeros_like(box_3d),
                    acc_frame=0)

        # Handle vanished tracklets
        for tid in self.tracklets:
            if cur_frame > self.tracklets[tid]['last_frame'] and tid > -1:
                self.tracklets[tid]['box_3d'][:self.loc_dim] = self.tracklets[
                    tid]['box_3d'].new_tensor(
                        self.tracklets[tid]['tracker'].predict()
                        [:self.loc_dim])

        # Add backdrops
        backdrop_inds = torch.nonzero(ids == -1).squeeze(1)
        ious = bbox_overlaps(bboxes[backdrop_inds, :-1], bboxes[:, :-1])
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        backdrop_tracker = [
            self.tracker_model(self.device, self.lstm,
                               boxes_3d[bd_ind].cpu().numpy(),
                               depth_uncertainty[bd_ind].cpu().numpy()) if
            self.tracker_model_name == 'LSTM3DTracker' else self.tracker_model(
                boxes_3d[bd_ind].cpu().numpy(),
                depth_uncertainty[bd_ind].cpu().numpy())
            for bd_ind in backdrop_inds
        ]

        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                boxes_3d=boxes_3d[backdrop_inds],
                tracker=backdrop_tracker,
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if cur_frame - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_boxes_3d = []
        memo_trackers = []
        memo_labels = []
        memo_vs = []
        for k, v in self.tracklets.items():
            memo_bboxes.append(v['bbox'][None, :])
            memo_boxes_3d.append(v['box_3d'][None, :])
            memo_trackers.append(v['tracker'])
            memo_embeds.append(v['embed'][None, :])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = torch.full((1, backdrop['embeds'].size(0)),
                                      -1,
                                      dtype=torch.long)
            backdrop_vs = torch.zeros_like(backdrop['boxes_3d'])
            memo_bboxes.append(backdrop['bboxes'])
            memo_boxes_3d.append(backdrop['boxes_3d'])
            memo_trackers.extend(backdrop['tracker'])
            memo_embeds.append(backdrop['embeds'])
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_labels.append(backdrop['labels'][:, None])
            memo_vs.append(backdrop_vs)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_boxes_3d = torch.cat(memo_boxes_3d, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_bboxes, memo_labels, memo_boxes_3d, memo_trackers, memo_embeds, memo_ids.squeeze(
            0), memo_vs

    def match(self,
              bboxes: torch.Tensor,
              labels: torch.Tensor,
              boxes_3d: torch.Tensor,
              depth_uncertainty: torch.Tensor,
              position: torch.Tensor,
              rotation: torch.Tensor,
              embeds: torch.Tensor,
              cur_frame: int,
              pure_det: bool = False):
        """Match incoming detection results with embedding and 3D infos

        Args:
            bboxes (torch.tensor): (N, 5), [x1, y1, x2, y2, conf]
            labels (torch.tensor): (N,)
            boxes_3d (torch.tensor): (N, 7), 3D information stored
                                     in world coordinates with the format
                                     [X, Y, Z, theta, h, w, l]
            depth_uncertainty (torch.tensor): (N, ), confidence in depth 
                                     estimation
            position (torch.tensor): (3, ), camera position
            rotation (torch.tensor): (3, 3), camera rotation
            embeds (torch.tensor): (N, C), extracted box feature
            cur_frame (int): indicates the frame index
            pure_det (bool): output pure detection. Defaults False.

        Raises:
            NotImplementedError: raise if self.match_metric not found

        Returns:
            list: A list of matched bbox, labels, boxes_3d and embeds
        """
        if depth_uncertainty is None or not self.with_depth_uncertainty:
            depth_uncertainty = boxes_3d.new_ones((boxes_3d.shape[0], 1))

        _, inds = (bboxes[:, -1] *
                   depth_uncertainty.flatten()).sort(descending=True)
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        embeds = embeds[inds, :]
        boxes_3d = boxes_3d[inds]
        depth_uncertainty = depth_uncertainty[inds]

        if pure_det:
            valids = bboxes.new_ones((bboxes.size(0)), dtype=torch.bool)
            ids = torch.arange(
                self.num_tracklets,
                self.num_tracklets + bboxes.size(0),
                dtype=torch.long)
            self.num_tracklets += bboxes.size(0)
            return bboxes, labels, boxes_3d, ids, inds, valids

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.size(0)):
            thr = self.nms_backdrop_iou_thr if bboxes[
                i, -1] < self.obj_score_thr else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        embeds = embeds[valids, :]
        boxes_3d = boxes_3d[valids]
        depth_uncertainty = depth_uncertainty[valids]

        # init ids container
        ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            memo_bboxes, memo_labels, memo_boxes_3d, \
                memo_trackers, memo_embeds, memo_ids, memo_vs = self.memo

            mmcv.check_accum_time('predict', counting=True)
            memo_boxes_3d_predict = memo_boxes_3d.detach().clone()
            for ind, memo_tracker in enumerate(memo_trackers):
                memo_velo = memo_tracker.predict(
                    update_state=memo_tracker.age != 0)
                memo_boxes_3d_predict[ind, :3] += memo_boxes_3d.new_tensor(
                    memo_velo[7:])
            mmcv.check_accum_time('predict', counting=False)

            if self.with_bbox_iou:

                def get_xy_box(boxes_3d_world):
                    box_x_cen = boxes_3d_world[:, 0]
                    box_y_cen = boxes_3d_world[:, 1]
                    box_width = boxes_3d_world[:, 5]
                    box_length = boxes_3d_world[:, 6]

                    dets_xy_box = torch.stack([
                        box_x_cen - box_width / 2.0, box_y_cen -
                        box_length / 2.0, box_x_cen + box_width / 2.0,
                        box_y_cen + box_length / 2.0
                    ],
                                              dim=1)
                    return dets_xy_box

                if self.track_bbox_iou == 'box2d':
                    scores_iou = bbox_overlaps(bboxes[:, :-1],
                                               memo_bboxes[:, :-1])
                elif self.track_bbox_iou == 'bev':
                    dets_xy_box = get_xy_box(boxes_3d)
                    memo_dets_xy_box = get_xy_box(memo_boxes_3d_predict)
                    scores_iou = bbox_overlaps(dets_xy_box, memo_dets_xy_box)
                elif self.track_bbox_iou == 'box3d':
                    depth_weight = F.pairwise_distance(
                        boxes_3d[..., None],
                        memo_boxes_3d_predict[..., None].transpose(2, 0))
                    scores_iou = torch.exp(-depth_weight / 10.0)
                elif self.track_bbox_iou == 'box2d_depth_aware':
                    depth_weight = F.pairwise_distance(
                        boxes_3d[..., None],
                        memo_boxes_3d_predict[..., None].transpose(2, 0))
                    scores_iou = torch.exp(-depth_weight / 10.0)
                    scores_iou *= bbox_overlaps(bboxes[:, :-1],
                                                memo_bboxes[:, :-1])
                else:
                    raise NotImplementedError
            else:
                scores_iou = bboxes.new_ones(
                    [bboxes.size(0), memo_bboxes.size(0)])

            if self.with_deep_feat:

                def compute_quasi_dense_feat_match(embeds, memo_embeds):
                    if self.match_metric == 'cycle_softmax':
                        feats = torch.mm(embeds, memo_embeds.t())
                        d2t_scores = feats.softmax(dim=1)
                        t2d_scores = feats.softmax(dim=0)
                        scores_feat = (d2t_scores + t2d_scores) / 2
                    elif self.match_metric == 'softmax':
                        feats = torch.mm(embeds, memo_embeds.t())
                        scores_feat = feats.softmax(dim=1)
                    elif self.match_metric == 'cosine':
                        scores_feat = torch.mm(
                            F.normalize(embeds, p=2, dim=1),
                            F.normalize(memo_embeds, p=2, dim=1).t())
                    else:
                        raise NotImplementedError
                    return scores_feat

                scores_feat = compute_quasi_dense_feat_match(
                    embeds, memo_embeds)
            else:
                scores_feat = scores_iou.new_ones(scores_iou.shape)

            # Match with depth ordering
            if self.with_depth_ordering:

                def compute_boxoverlap_with_depth(obsv_boxes_3d, memo_boxes_3d,
                                                  memo_vs):
                    # Sum up all the available region of each tracker
                    if self.depth_match_metric == 'centroid':
                        depth_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., :3, None],
                            memo_boxes_3d[..., :3, None].transpose(2, 0))
                        depth_weight = torch.exp(-depth_weight / 10.0)
                    elif self.depth_match_metric == 'cosine':
                        match_corners_observe = tu.worldtocamera_torch(
                            obsv_boxes_3d[:, :3], position, rotation)
                        match_corners_predict = tu.worldtocamera_torch(
                            memo_boxes_3d[:, :3], position, rotation)
                        depth_weight = F.cosine_similarity(
                            match_corners_observe[..., None],
                            match_corners_predict[..., None].transpose(2, 0))
                        depth_weight += 1.0
                        depth_weight /= 2.0
                    elif self.depth_match_metric == 'pure_motion':
                        # Moving distance should be aligned
                        # V_observed-tracked vs. V_velocity
                        depth_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., :3, None] -
                            memo_boxes_3d[..., :3, None].transpose(2, 0),
                            memo_vs[..., :3, None].transpose(2, 0))
                        depth_weight = torch.exp(-depth_weight / 5.0)
                        # Moving direction should be aligned
                        # Set to 0.5 when two vector not within +-90 degree
                        cos_sim = F.cosine_similarity(
                            obsv_boxes_3d[..., :2, None] -
                            memo_boxes_3d[..., :2, None].transpose(2, 0),
                            memo_vs[..., :2, None].transpose(2, 0))
                        cos_sim += 1.0
                        cos_sim /= 2.0
                        depth_weight *= cos_sim
                    elif self.depth_match_metric == 'motion':
                        centroid_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., :3, None],
                            memo_boxes_3d_predict[..., :3,
                                                  None].transpose(2, 0))
                        centroid_weight = torch.exp(-centroid_weight / 10.0)
                        # Moving distance should be aligned
                        # V_observed-tracked vs. V_velocity
                        motion_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., :3, None] -
                            memo_boxes_3d[..., :3, None].transpose(2, 0),
                            memo_vs[..., :3, None].transpose(2, 0))
                        motion_weight = torch.exp(-motion_weight / 5.0)
                        # Moving direction should be aligned
                        # Set to 0.5 when two vector not within +-90 degree
                        cos_sim = F.cosine_similarity(
                            obsv_boxes_3d[..., :2, None] -
                            memo_boxes_3d[..., :2, None].transpose(2, 0),
                            memo_vs[..., :2, None].transpose(2, 0))
                        cos_sim += 1.0
                        cos_sim /= 2.0
                        depth_weight = cos_sim * centroid_weight + (
                            1.0 - cos_sim) * motion_weight
                    else:
                        raise NotImplementedError

                    return depth_weight

                if self.depth_match_metric == 'motion':
                    scores_depth = compute_boxoverlap_with_depth(
                        boxes_3d, memo_boxes_3d, memo_vs)
                else:
                    scores_depth = compute_boxoverlap_with_depth(
                        boxes_3d, memo_boxes_3d_predict, memo_vs)
            else:
                scores_depth = scores_iou.new_ones(scores_iou.shape)

            if self.with_cats:
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                scores_cats = cat_same.float()
            else:
                scores_cats = scores_iou.new_ones(scores_iou.shape)

            scores = self.bbox_affinity_weight * scores_iou * scores_depth + \
                self.feat_affinity_weight * scores_feat
            scores /= (self.bbox_affinity_weight + self.feat_affinity_weight)
            scores *= (scores_iou > 0.0).float()
            scores *= (scores_depth > 0.0).float()
            scores *= scores_cats

            # Assign matching
            mmcv.check_accum_time('matching', counting=True)
            if self.match_algo == 'greedy':
                for i in range(bboxes.size(0)):
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                    tid = memo_ids[memo_ind]
                    # Matching confidence
                    if conf > self.match_score_thr:
                        # Update existing tracklet
                        if tid > -1:
                            # Keep object with high 3D objectness
                            if bboxes[i, -1] * depth_uncertainty[
                                    i] > self.obj_score_thr:
                                ids[i] = tid
                                scores[:i, memo_ind] = 0
                                scores[i + 1:, memo_ind] = 0
                            else:
                                # Reduce FP w/ low objectness but high match conf
                                if conf > self.nms_conf_thr:
                                    ids[i] = -2
            elif self.match_algo == 'hungarian':
                # Hungarian
                matched_indices = linear_assignment(-scores.cpu().numpy())
                for idx in range(len(matched_indices[0])):
                    i = matched_indices[0][idx]
                    memo_ind = matched_indices[1][idx]
                    conf = scores[i, memo_ind]
                    tid = memo_ids[memo_ind]
                    if conf > self.match_score_thr and tid > -1:
                        # Keep object with high 3D objectness
                        if bboxes[i, -1] * depth_uncertainty[
                                i] > self.obj_score_thr:
                            ids[i] = tid
                            scores[:i, memo_ind] = 0
                            scores[i + 1:, memo_ind] = 0
                        else:
                            # Reduce FP w/ low objectness but high match conf
                            if conf > self.nms_conf_thr:
                                ids[i] = -2
                del matched_indices
            mmcv.check_accum_time('matching', counting=False)

            if self._debug:
                bbox_inds = scores.max(1).indices
                print("\nTotal:\n"
                      f"Scores: {scores.max(1).values.cpu().numpy()}\n"
                      f"Indices: {scores.max(1).indices.cpu().numpy()}\n"
                      "IoU:\n"
                      f"Scores: {scores_iou.max(1).values.cpu().numpy()}\n"
                      f"Indices: {scores_iou.max(1).indices.cpu().numpy()}\n"
                      f"Feat:\n"
                      f"Scores: {scores_feat.max(1).values.cpu().numpy()}\n"
                      f"Indices: {scores_feat.max(1).indices.cpu().numpy()}\n"
                      f"Depth:\n"
                      f"Scores: {scores_depth.max(1).values.cpu().numpy()}\n"
                      f"Indices: {scores_depth.max(1).indices.cpu().numpy()}")
                print("Uncertainty:\n",
                      depth_uncertainty.flatten().cpu().numpy())
                print(f"Memo: {memo_boxes_3d.shape}\tMemo Ids: {memo_ids}\n"
                      f"{memo_boxes_3d[bbox_inds].cpu().numpy()}")
                print(f"Velo:\n{memo_vs[bbox_inds].cpu().numpy()}")
                print(
                    f"Pred:\n{memo_boxes_3d_predict[bbox_inds].cpu().numpy()}")
                print(f"Obsv: {boxes_3d.shape}\tObsv Ids: {ids}\n"
                      f"{boxes_3d.cpu().numpy()}")

        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_news,
            dtype=torch.long)
        self.num_tracklets += num_news

        self.update_memo(ids, bboxes, boxes_3d, depth_uncertainty, embeds,
                         labels, cur_frame)

        update_bboxes = bboxes.detach().clone()
        update_labels = labels.detach().clone()
        update_boxes_3d = boxes_3d.detach().clone()
        for tid in ids[ids > -1]:
            update_boxes_3d[ids == tid] = self.tracklets[int(tid)]['box_3d']
        update_ids = ids.detach().clone()

        if self._debug:
            print(
                f"Updt: {update_boxes_3d.shape}\tUpdt ID: {update_ids.cpu().numpy()}\n"
                f"{update_boxes_3d.cpu().numpy()}")

        return update_bboxes, update_labels, update_boxes_3d, update_ids, inds, valids
