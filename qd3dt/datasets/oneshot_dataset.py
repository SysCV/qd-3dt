import os.path as osp
import random
import numpy as np
from collections import defaultdict

from pycocotools.coco import COCO

import mmcv
from mmcv.parallel import DataContainer as DC

from qd3dt.datasets import CocoDataset
from qd3dt.datasets.utils import to_tensor, random_scale
from qd3dt.datasets.registry import DATASETS
from qd3dt.datasets.auto_augment import auto_augment
from qd3dt.apis import get_root_logger


@DATASETS.register_module
class OneshotDataset(CocoDataset):

    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa'
    ]

    NOVEL_CLASSES = {
        1: ['bird', 'bus', 'cow', 'motorbike', 'sofa'],
        2: ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
        3: ['boat', 'cat', 'motorbike', 'sheep', 'sofa'],
    }

    def __init__(self, num_ref_imgs=4, *args, **kwargs):
        super(OneshotDataset, self).__init__(*args, **kwargs)
        self.NOVEL_CLASSES = self.NOVEL_CLASSES[self.novel_set]
        self.label2cat = {v: k for k, v in self.cat2label.items()}
        self.num_ref_imgs = num_ref_imgs

    def sample_support(self):
        pass

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        ann = self.get_ann_info(idx=idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']
        if len(gt_bboxes) == 0:
            return None
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        cat_ids = self.coco.getCatIds()
        key_cat_ids = []
        for k in set(gt_labels):
            key_cat_ids.append(self.label2cat[k])
        if len(key_cat_ids) >= self.num_ref_imgs:
            ref_cat_ids = random.choices(key_cat_ids, k=self.num_ref_imgs)
        else:
            ref_cat_ids = key_cat_ids.copy()
            for k in ref_cat_ids:
                cat_ids.pop(cat_ids.index(k))
            ref_cat_ids.extend(
                random.sample(cat_ids, self.num_ref_imgs - len(key_cat_ids)))
        assert len(ref_cat_ids) == self.num_ref_imgs

        ref_img_ids = []
        for i, cat_id in enumerate(ref_cat_ids):
            img_ids = self.coco.getImgIds(catIds=cat_id)
            img_id = random.choice(img_ids)
            ref_img_ids.append(img_id)

        ref_imgs = []
        ref_img_metas = []
        ref_bboxes = []
        ref_labels = []
        ref_num_gts = []

        all_ref_cats = ref_cat_ids.copy()
        for i, (img_id, cat_id) in enumerate(zip(ref_img_ids, ref_cat_ids)):
            _bboxes = []
            _labels = []

            ref_img_info = self.coco.loadImgs(img_id)[0]
            _img = mmcv.imread(
                osp.join(self.img_prefix, ref_img_info['filename']))
            _flip = True if np.random.rand() < self.flip_ratio else False
            _img, _img_shape, _pad_shape, _scale_factor = self.img_transform(
                _img, img_scale, _flip, keep_ratio=self.resize_keep_ratio)

            ann_ids = self.coco.getAnnIds(img_id)
            ann_infos = self.coco.loadAnns(ann_ids)

            maps = defaultdict(list)
            for ann_info in ann_infos:
                maps[ann_info['category_id']].append(ann_info['id'])

            for k, v in maps.items():
                if k == cat_id or k not in all_ref_cats:
                    all_ref_cats.append(k)
                    ann_id = random.choice(v)
                    ann_info = self.coco.loadAnns(ann_id)[0]
                    x1, y1, w, h = ann_info['bbox']
                    bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                    _bboxes.append(bbox)
                    _labels.append(self.cat2label[k])

            _bboxes = np.array(_bboxes, dtype=np.float32)
            _bboxes = self.bbox_transform(_bboxes, _img_shape, _scale_factor,
                                          _flip)
            ref_img_meta = dict(
                ori_shape=(ref_img_info['height'], ref_img_info['width'], 3),
                img_shape=_img_shape,
                pad_shape=_pad_shape,
                scale_factor=_scale_factor,
                flip=_flip)

            ref_imgs.append(_img)
            ref_img_metas.append(ref_img_meta)
            ref_bboxes.append(_bboxes)
            ref_labels.append(np.array(_labels, dtype=np.int64))
            ref_num_gts.append(len(_labels))

        # ref_num_gts = np.array(ref_num_gts, dtype=np.int64)
        ref_bboxes = np.concatenate(ref_bboxes, axis=0)
        ref_labels = np.concatenate(ref_labels, axis=0)

        max_h = max([_img.shape[1] for _img in ref_imgs])
        max_w = max([_img.shape[2] for _img in ref_imgs])
        all_ref_imgs = np.zeros((len(ref_imgs), 3, max_h, max_w),
                                dtype=np.float)
        for i, ref_img in enumerate(ref_imgs):
            all_ref_imgs[i, :, :ref_img.shape[1], :ref_img.shape[2]] = ref_img

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            gt_labels=DC(to_tensor(gt_labels)),
            gt_bboxes_ignore=DC(to_tensor(gt_bboxes_ignore)),
            ref_imgs=DC(to_tensor(all_ref_imgs), stack=True),
            ref_img_metas=DC(ref_img_metas, cpu_only=True),
            ref_bboxes=DC(to_tensor(ref_bboxes)),
            ref_labels=DC(to_tensor(ref_labels)),
            ref_num_gts=DC(ref_num_gts, cpu_only=True))

        return data
