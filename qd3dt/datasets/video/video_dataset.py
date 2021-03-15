import os.path as osp
import random
import numpy as np
from collections import defaultdict

from .video_parser import VID
from pycocotools.coco import COCO

import mmcv
from mmcv.parallel import DataContainer as DC

from qd3dt.datasets import CustomDataset
from qd3dt.datasets.utils import to_tensor, random_scale
from qd3dt.datasets.registry import DATASETS
from qd3dt.datasets.auto_augment import auto_augment
from qd3dt.apis import get_root_logger

# import mc
import io
import cv2


def mc_loader(filename, mclient):
    # keep the value of the image the same as that from mmcv
    value = mc.pyvector()
    mclient.Get(filename, value)
    value_str = mc.ConvertBuffer(value)
    buff = io.BytesIO(value_str)
    img = cv2.imdecode(
        np.asarray(bytearray(buff.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


@DATASETS.register_module
class VideoDataset(CustomDataset):

    def __init__(self,
                 filter_empty_gt=True,
                 key_sample_interval=1,
                 ref_sample_interval=1,
                 ref_sample_sigma=-1,
                 track_det_ratio=-1,
                 ref_share_flip=False,
                 augs_num=1,
                 augs_ratio=0.7,
                 use_mc=False,
                 *args,
                 **kwargs):
        assert not use_mc
        self.logger = get_root_logger()
        # set the interval to sample key frames from a video
        assert key_sample_interval >= 1
        self.key_sample_interval = key_sample_interval
        # set the interval to sample ref frames for training
        assert ref_sample_interval >= 1
        self.ref_sample_interval = ref_sample_interval
        self.ref_sample_sigma = ref_sample_sigma
        # adjust the number of training images in **DET** set
        self.track_det_ratio = track_det_ratio
        self.filter_empty_gt = filter_empty_gt
        super(VideoDataset, self).__init__(*args, **kwargs)
        assert self.proposals is None
        assert not self.with_mask
        self.ref_share_flip = ref_share_flip
        self.augs_num = augs_num
        self.augs_ratio = augs_ratio
        self.use_mc = use_mc

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        if self.num_vid_imgs > 0:
            vvid_img_ids = set(_['image_id'] for _ in self.vid.anns.values())
            for i, img_info in enumerate(self.img_infos[:self.num_vid_imgs]):
                if self.filter_empty_gt and (self.img_ids[i]
                                             not in vvid_img_ids):
                    continue
                valid_inds.append(i)

        if self.num_det_imgs > 0:
            vdet_img_ids = set(_['image_id'] for _ in self.coco.anns.values())
            for i, img_info in enumerate(self.img_infos[self.num_vid_imgs:]):
                if self.filter_empty_gt and self.img_ids[
                        i + self.num_vid_imgs] not in vdet_img_ids:
                    continue
                valid_inds.append(i + self.num_vid_imgs)
        return valid_inds

    def sample_key_frames(self, vid_id):
        img_ids = self.vid.getImgIdsFromVidId(vid_id)
        if not self.test_mode:
            img_ids = img_ids[::self.key_sample_interval]
        return img_ids

    def load_annotations(self, ann_file):
        img_infos = []
        self.img_ids = []
        if isinstance(ann_file, str):
            ann_file = dict(VID=ann_file)

        if 'VID' in ann_file.keys():
            self.vid = VID(ann_file['VID'])
            self.vid_ids = self.vid.getVidIds()
            for vid_id in self.vid_ids:
                img_ids = self.sample_key_frames(vid_id)
                self.img_ids.extend(img_ids)
                for img_id in img_ids:
                    info = self.vid.loadImgs([img_id])[0]
                    info['filename'] = info['file_name']
                    info['type'] = 'VID'
                    info['first_frame'] = True if info['index'] == 0 else False
                    img_infos.append(info)
        self.num_vid_imgs = len(img_infos)

        if 'DET' in ann_file.keys():
            self.coco = COCO(ann_file['DET'])
            img_ids = self.coco.getImgIds()
            if self.track_det_ratio > 0:
                num_det_imgs = self.num_vid_imgs // self.track_det_ratio
                if len(img_ids) > num_det_imgs:
                    img_ids = img_ids[:num_det_imgs]
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.loadImgs([img_id])[0]
                info['filename'] = info['file_name']
                info['type'] = 'DET'
                img_infos.append(info)
        self.num_det_imgs = len(img_infos) - self.num_vid_imgs

        api = getattr(self, 'coco',
                      None) if self.num_vid_imgs == 0 else self.vid
        self.cat_ids = api.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        mode = 'TRAINING' if not self.test_mode else 'TESTING'
        msg = '**{}**: {} vs. {} images from tracking and detection set.'.format(
            mode, self.num_vid_imgs, self.num_det_imgs)
        self.logger.info(msg)
        return img_infos

    def random_ref_scale(self, img_scale):
        lower_img_scale = tuple(
            [int(s * (self.ref_scale_ratio)) for s in img_scale])
        upper_img_scale = tuple(
            [int(s * (2 - self.ref_scale_ratio)) for s in img_scale])
        ref_img_scales = [lower_img_scale, upper_img_scale]
        return random_scale(ref_img_scales, self.multiscale_mode)

    def get_ann_info(self, idx=None, has_ids=True, img_info=None):
        api = self.vid if has_ids else self.coco
        if img_info is None:
            img_id = self.img_infos[idx]['id']
        else:
            img_id = img_info['id']
        ann_ids = api.getAnnIds(imgIds=[img_id])
        anns = api.loadAnns(ann_ids)
        return self._parse_ann_info(anns, with_track=has_ids)

    def get_ref_policies(self, policies):
        ref_policies = []
        for policy in policies:
            if policy['share']:
                ref_policies.append(policy)
        no_share_policies = [p for p in self.aug_policies if not p['share']]
        ps = np.random.choice(
            no_share_policies,
            self.augs_num - len(ref_policies),
            replace=False)
        ref_policies.extend(ps)
        return ref_policies

    def get_aug_policies(self):
        augs = dict()
        ref_augs = dict()
        # scale
        augs['img_scale'] = random_scale(self.img_scales, self.multiscale_mode)
        ref_augs['img_scale'] = augs['img_scale']
        # flip
        augs['flip'] = True if np.random.rand() < self.flip_ratio else False
        if self.ref_share_flip:
            ref_augs['flip'] = augs['flip']
        else:
            ref_augs['flip'] = True if np.random.rand(
            ) < self.flip_ratio else False
        # other augs
        if self.aug_policies is not None and np.random.rand(
        ) < self.augs_ratio:
            policies = np.random.choice(
                self.aug_policies, self.augs_num, replace=False)
            augs['auto_aug'] = policies
            ref_augs['auto_aug'] = self.get_ref_policies(policies)
        return augs, ref_augs

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        has_ids = img_info['type'] == 'VID'
        ann = self.get_ann_info(idx=idx, has_ids=has_ids)
        ref_img_info = self.sample_ref_img(img_info, has_ids)
        ref_ann = self.get_ann_info(img_info=ref_img_info, has_ids=has_ids)
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
        return data

    def transform(self, img_info, ann, augs):
        if self.use_mc:
            if not hasattr(self, 'mclient'):
                server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
                client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
                self.mclient = mc.MemcachedClient.GetInstance(
                    server_list_config_file, client_config_file)
            img = mc_loader(img_info['filename'], self.mclient)
        else:
            img = mmcv.imread(img_info['filename'])
        ori_shape = (img_info['height'], img_info['width'], 3)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        gt_bboxes_ignore = ann['bboxes_ignore']

        if 'auto_aug' in augs.keys():
            img, gt_bboxes = auto_augment(img, gt_bboxes, augs['auto_aug'])

        flip = augs['flip']
        img_scale = augs['img_scale']
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                               scale_factor, flip)
        calib = img_info['cali']
        pose = img_info['pose']
        fov = img_info['fov']
        near_clip = img_info['near_clip']
        if ori_shape != img_shape:
            focal_length = calib[0][0]
            width = img_shape[1]
            height = img_shape[0]
            calib = [[focal_length * scale_factor, 0, width / 2.0, 0],
                     [0, focal_length * scale_factor, height / 2.0, 0],
                     [0, 0, 1, 0]]

        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            calib=calib,
            pose=pose,
            fov=fov,
            near_clip=near_clip,
            scale_factor=scale_factor,
            img_info=img_info,
            flip=flip)

        if 'instance_ids' in ann.keys():
            img_meta['instance_ids'] = ann['instance_ids']

        return img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore

    def sample_ref_img(self, img_info, has_ids=True):
        if has_ids:
            vid_id = img_info['video_id']
            img_ids = self.vid.getImgIdsFromVidId(vid_id)
            index = img_info['index']
            if self.ref_sample_sigma > -1:
                offset = np.random.normal(
                    0, self.ref_sample_sigma) * self.ref_sample_interval
                offset = np.ceil(offset) if offset > 0 else np.floor(offset)
                ref_index = min(max(index + offset, len(img_ids) - 1), 0)
                ref_img_id = img_ids[ref_index]
            else:
                left = max(0, index - self.ref_sample_interval)
                right = min(index + self.ref_sample_interval, len(img_ids) - 1)
                valid_inds = img_ids[left:index] + img_ids[index + 1:right + 1]
                ref_img_id = random.choice(valid_inds)
            ref_img_info = self.vid.loadImgs([ref_img_id])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
            ref_img_info['type'] = 'VID'
        else:
            ref_img_info = img_info.copy()
        return ref_img_info

    def matching(self, ann, ref_ann):
        if 'instance_ids' in ann.keys():
            gt_instances = ann['instance_ids']
            ref_instances = ref_ann['instance_ids']
            gt_pids = [
                ref_instances.index(i) if i in ref_instances else -1
                for i in gt_instances
            ]
            ref_gt_pids = [
                gt_instances.index(i) if i in gt_instances else -1
                for i in ref_instances
            ]
        else:
            gt_pids = np.arange(ann['bboxes'].shape[0], dtype=np.int64)
            ref_gt_pids = gt_pids.copy()
        return gt_pids, ref_gt_pids

    def _parse_ann_info(self, ann_info, with_track=True):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
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

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            labels_ignore=gt_labels_ignore)
        if with_track:
            ann['instance_ids'] = gt_instances
        return ann

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['file_name']))
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
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
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
