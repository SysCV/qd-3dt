import numpy as np
import os.path as osp

import mmcv
from mmcv.parallel import DataContainer as DC
from pycocotools.coco import COCO

from .video_dataset import VideoDataset
from .video_parser import VID
from qd3dt.datasets.registry import DATASETS
from qd3dt.datasets.utils import to_tensor


@DATASETS.register_module
class BDDVid3DDataset(VideoDataset):

    CLASSES = ('person', 'vehicle', 'bike')

    def __init__(self, **kwargs):
        super(BDDVid3DDataset, self).__init__(**kwargs)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        has_ids = img_info['type'] == 'VID'
        # TODO: check has_3d vs with_3d from config
        has_3d = self.with_3d
        ann = self.get_ann_info(idx=idx, has_ids=has_ids, has_3d=has_3d)
        ref_img_info = self.sample_ref_img(img_info, has_ids=has_ids)
        ref_ann = self.get_ann_info(
            img_info=ref_img_info, has_ids=has_ids, has_3d=has_3d)
        if len(ann['bboxes']) == 0 or len(ref_ann['bboxes']) == 0:
            return None
        gt_pids, ref_gt_pids = self.matching(ann, ref_ann)
        if sum(gt_pids) == -1 * len(gt_pids):
            return None
        augs, ref_augs = self.get_aug_policies()
        img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore = self.transform(
            img_info, ann, augs)
        (ref_img, ref_img_meta, ref_gt_bboxes, ref_gt_labels,
         ref_gt_bboxes_ignore) = self.transform(ref_img_info, ref_ann,
                                                ref_augs)

        if self.with_track:
            data = dict(
                img=DC(to_tensor(img), stack=True),
                img_meta=DC(img_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)),
                gt_labels=DC(to_tensor(gt_labels)),
                gt_bboxes_ignore=DC(to_tensor(gt_bboxes_ignore)),
                gt_pids=DC(to_tensor(gt_pids)),
                ref_img=DC(to_tensor(ref_img), stack=True),
                ref_img_meta=DC(ref_img_meta, cpu_only=True),
                ref_gt_bboxes=DC(to_tensor(ref_gt_bboxes)),
                ref_gt_labels=DC(to_tensor(ref_gt_labels)),
                ref_gt_bboxes_ignore=DC(to_tensor(ref_gt_bboxes_ignore)),
                ref_gt_pids=DC(to_tensor(ref_gt_pids)))
        else:
            data = dict(
                img=DC(to_tensor(img), stack=True),
                img_meta=DC(img_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)),
                gt_labels=DC(to_tensor(gt_labels)),
                gt_bboxes_ignore=DC(to_tensor(gt_bboxes_ignore)))

        if has_3d:
            if augs['flip']:
                alpha_mask = ann['alpha'] > 0
                ann['alpha'][alpha_mask] = np.pi - ann['alpha'][alpha_mask]
                ann['alpha'][~alpha_mask] = -np.pi - ann['alpha'][~alpha_mask]
                # cen_2ds is delta x, y from box center
                ann['cen_2ds'][:, 0] = -ann['cen_2ds'][:, 0]
            if ref_augs['flip']:
                alpha_mask = ref_ann['alpha'] > 0
                ref_ann['alpha'][
                    alpha_mask] = np.pi - ref_ann['alpha'][alpha_mask]
                ref_ann['alpha'][
                    ~alpha_mask] = -np.pi - ref_ann['alpha'][~alpha_mask]
                ref_ann['cen_2ds'][:, 0] = -ref_ann['cen_2ds'][:, 0]

            data['gt_alphas'] = DC(to_tensor(ann['alpha']))
            data['ref_gt_alphas'] = DC(to_tensor(ref_ann['alpha']))
            data['gt_rotys'] = DC(to_tensor(ann['roty']))
            data['ref_gt_rotys'] = DC(to_tensor(ref_ann['roty']))
            data['gt_trans'] = DC(to_tensor(ann['trans']))
            data['ref_gt_trans'] = DC(to_tensor(ref_ann['trans']))
            data['gt_dims'] = DC(to_tensor(ann['dims']))
            data['ref_gt_dims'] = DC(to_tensor(ref_ann['dims']))
            data['gt_2dcs'] = DC(to_tensor(ann['cen_2ds']))
            data['ref_gt_2dcs'] = DC(to_tensor(ref_ann['cen_2ds']))

        return data

    def get_ann_info(self, idx=None, has_ids=True, has_3d=True, img_info=None):
        api = self.vid if has_ids else self.coco
        if img_info is None:
            img_id = self.img_infos[idx]['id']
        else:
            img_id = img_info['id']
        ann_ids = api.getAnnIds(imgIds=[img_id])
        anns = api.loadAnns(ann_ids)
        return self._parse_ann_info(anns, with_track=has_ids, with_3d=has_3d)

    def _parse_ann_info(self, ann_info, with_track=True, with_3d=True):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        if with_3d:
            gt_alphas = []
            gt_rotys = []
            gt_dims = []
            gt_trans = []
            gt_2dcs = []
        if with_track:
            gt_instances = []

        for i, ann in enumerate(ann_info):
            if ann.get('iscrowd', False):
                ann['ignore'] = False
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
                gt_labels_ignore.append(self.cat2label[ann['category_id']])
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if with_3d:
                    gt_alphas.append(ann['alpha'])
                    gt_rotys.append(ann['roty'])
                    gt_dims.append(ann['dimension'])
                    gt_trans.append(ann['translation'])
                    gt_2dcs.append(ann['delta_2d'])
                if with_track:
                    gt_instances.append(ann['instance_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore and self.with_crowd:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            gt_labels_ignore = np.array(gt_labels_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            gt_labels_ignore = np.zeros((0, ))

        if with_3d:
            gt_alphas = np.array(gt_alphas, dtype=np.float32)
            gt_rotys = np.array(gt_rotys, dtype=np.float32)
            gt_dims = np.array(gt_dims, dtype=np.float32).reshape(-1, 3)
            gt_trans = np.array(gt_trans, dtype=np.float32).reshape(-1, 3)
            gt_2dcs = np.array(gt_2dcs, dtype=np.float32).reshape(-1, 2)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            labels_ignore=gt_labels_ignore)
        if with_3d:
            ann['alpha'] = gt_alphas
            ann['roty'] = gt_rotys
            ann['dims'] = gt_dims
            ann['trans'] = gt_trans
            ann['cen_2ds'] = gt_2dcs
        if with_track:
            ann['instance_ids'] = gt_instances

        return ann

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        if img_info['type'] == 'DET':
            img_info['first_frame'] = True
        img = mmcv.imread(img_info['file_name'])

        # TODO: Not support props now. Need to re-index if include props
        proposal = None

        if img is None:
            print(img_info['file_name'], flush=True)
            import pdb
            pdb.set_trace()

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            calib = img_info['cali']
            pose = img_info['pose']
            fov = img_info['fov']
            near_clip = img_info['near_clip']
            ori_shape = (img_info['height'], img_info['width'], 3)

            if ori_shape != img_shape:
                focal_length = calib[0][0]
                width = img_shape[1]
                height = img_shape[0]
                calib = [[focal_length * scale_factor, 0, width / 2.0, 0],
                         [0, focal_length * scale_factor, height / 2.0, 0],
                         [0, 0, 1, 0]]

            _img_meta = dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                calib=calib,
                pose=pose,
                fov=fov,
                near_clip=near_clip,
                first_frame=img_info['first_frame'],
                video_id=img_info['video_id'],
                frame_id=img_info['index'],
                scale_factor=scale_factor,
                img_info=img_info,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack([_proposal, score
                                       ]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        return data
