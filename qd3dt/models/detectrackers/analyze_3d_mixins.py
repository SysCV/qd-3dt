import os
import os.path as osp
import numpy as np
import torch
import mmcv
import platform

import scripts.tracking_utils as tu

if platform.system() == 'Linux':
    from qd3dt.core import bbox_overlaps
    from .utils.visual import imshow_3d_tracklets, imshow_bboxes_w_ids
else:
    from qd3dt.core.bbox.geometry import bbox_overlaps


class Analyze3DMixin(object):

    def analyze(self,
                img_meta,
                bboxes,
                labels,
                ids,
                depths=None,
                dims=None,
                alphas=None,
                cen_2ds=None,
                show=False,
                save=False,
                gt_cats=None):
        gt_bboxes, gt_labels, gt_ids, gt_ignores, \
            gt_alphas, gt_rotys, gt_dims, gt_trans, gt_2dcs = self.loadGts(
                img_meta, gt_cats)
        track_inds = ids > -1
        track_bboxes = bboxes[track_inds]
        track_labels = labels[track_inds]
        if depths is not None:
            track_depths = depths[track_inds]
        else:
            track_depths = None
        if dims is not None:
            track_dims = dims[track_inds]
        else:
            track_dims = None
        if alphas is not None:
            track_alphas = alphas[track_inds]
        else:
            track_alphas = None
        if cen_2ds is not None:
            track_2dcs = cen_2ds[track_inds]
        else:
            track_2dcs = None
        track_ids = ids[track_inds]
        if len(gt_ignores) > 0:
            ignore_inds = (bbox_overlaps(
                bboxes[:, :4], gt_ignores, mode='iof') > 0.5).any(dim=1)
        if track_bboxes.size(0) == 0:
            self.counter.num_fn += gt_bboxes.size(0)
            return
        if gt_bboxes.size(0) == 0:
            self.counter.num_fp += track_bboxes.size(0)
            if gt_ignores.size(0) > 0:
                self.counter.num_fp -= ignore_inds[track_inds].sum()
            return
        # init
        # [N, 6]: [x1, y1, x2, y2, class, id]
        self.counter.num_gt += gt_bboxes.size(0)
        fps = torch.ones(bboxes.size(0), dtype=torch.long)
        fns = torch.ones(gt_bboxes.size(0), dtype=torch.long)
        # false negatives after tracking filter
        track_fns = torch.ones(gt_bboxes.size(0), dtype=torch.long)
        idsw = torch.zeros(track_ids.size(0), dtype=torch.long)

        # fp & fn for raw detection results
        ious = bbox_overlaps(bboxes[:, :4], gt_bboxes[:, :4])
        same_cat = labels.view(-1, 1) == gt_labels.view(1, -1)
        ious *= same_cat.float()
        max_ious, gt_inds = ious.max(dim=1)
        _, dt_inds = bboxes[:, -1].sort(descending=True)
        for dt_ind in dt_inds:
            iou, gt_ind = max_ious[dt_ind], gt_inds[dt_ind]
            if iou > 0.5 and fns[gt_ind] == 1:
                fns[gt_ind] = 0
                if ids[dt_ind] > -1:
                    track_fns[gt_ind] = 0
                gt_bboxes[gt_ind, 4] = bboxes[dt_ind, -1]
                fps[dt_ind] = 0
            else:
                if len(gt_ignores) > 0 and ignore_inds[dt_ind]:
                    fps[dt_ind] = 0
                    gt_inds[dt_ind] = -2
                else:
                    gt_inds[dt_ind] = -1

        track_gt_inds = gt_inds[track_inds]
        track_fps = fps[track_inds]

        for i, tid in enumerate(track_ids):
            tid = int(tid)
            gt_ind = track_gt_inds[i]
            if gt_ind == -1 or gt_ind == -2:
                continue
            gt_id = int(gt_ids[gt_ind])
            if gt_id in self.id_maps.keys() and self.id_maps[gt_id] != tid:
                idsw[i] = 1
            if gt_id not in self.id_maps.keys() and tid in self.id_maps.values(
            ):
                idsw[i] = 1
            self.id_maps[gt_id] = tid

        fp_inds = track_fps == 1
        fn_inds = track_fns == 1
        idsw_inds = idsw == 1
        self.counter.num_fp += fp_inds.sum()
        self.counter.num_fn += fn_inds.sum()
        self.counter.num_idsw += idsw_inds.sum()

        if show or save:
            vid_name = os.path.dirname(
                img_meta[0]['img_info']['file_name']).split('/')[-1]
            img_name = os.path.basename(img_meta[0]['img_info']['file_name'])
            # img = os.path.join(
            #     self.data.img_prefix[img_meta[0]['img_info']['type']],
            #     vid_name, img_name)
            img = img_meta[0]['img_info']['file_name']
            save_path = os.path.join(self.out, 'analysis', vid_name)
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, img_name) if save else None
            img = imshow_3d_tracklets(
                img,
                track_bboxes[fp_inds].numpy(),
                track_labels[fp_inds].numpy(),
                depths=track_depths[fp_inds].numpy()
                if depths is not None else None,
                cen_2d=track_2dcs[fp_inds].numpy()
                if cen_2ds is not None else None,
                ids=track_ids[fp_inds].numpy(),
                color='red',
                show=False)
            img = imshow_3d_tracklets(
                img,
                gt_bboxes[fn_inds, :].numpy(),
                gt_labels[fn_inds].numpy(),
                depths=gt_trans[fn_inds, -1].numpy(),
                cen_2d=gt_2dcs[fn_inds, -1].numpy(),
                ids=gt_ids[fn_inds].numpy(),
                color='yellow',
                show=False)
            img = imshow_3d_tracklets(
                img,
                track_bboxes[idsw_inds].numpy(),
                track_labels[idsw_inds].numpy(),
                depths=track_depths[idsw_inds].numpy()
                if depths is not None else None,
                cen_2d=track_2dcs[idsw_inds].numpy()
                if cen_2ds is not None else None,
                ids=track_ids[idsw_inds].numpy(),
                color='cyan',
                show=show,
                out_file=save_file)

    def loadGts(self, img_meta, gt_cats=None):
        vid = self.dataset.vid if img_meta[0]['img_info'][
            'type'] == 'VID' else self.dataset.coco
        img_id = img_meta[0]['img_info']['id']
        ann_ids = vid.getAnnIds(img_id)
        anns = vid.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_alphas = []
        gt_rotys = []
        gt_dims = []
        gt_trans = []
        gt_2dcs = []
        gt_ignores = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            if gt_cats is not None and ann['category_id'] not in gt_cats:
                continue
            if ann['iscrowd'] or ann['ignore']:
                gt_ignores.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(ann['category_id'] - 1)
                gt_ids.append(ann['instance_id'])
                gt_alphas.append(ann['alpha'])
                gt_rotys.append(ann['roty'])
                gt_dims.append(ann['dimension'])
                gt_trans.append(ann['translation'])
                gt_2dcs.append(ann['center_2d'])
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float)
        gt_bboxes = torch.cat((gt_bboxes, torch.zeros(gt_bboxes.size(0), 1)),
                              dim=1)
        gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        gt_ids = torch.tensor(gt_ids, dtype=torch.long)
        gt_ignores = torch.tensor(gt_ignores, dtype=torch.float)
        gt_alphas = torch.tensor(gt_alphas, dtype=torch.float)
        gt_rotys = torch.tensor(gt_rotys, dtype=torch.float)
        gt_dims = torch.tensor(gt_dims, dtype=torch.float)
        gt_trans = torch.tensor(gt_trans, dtype=torch.float)
        gt_2dcs = torch.tensor(gt_2dcs, dtype=torch.float)
        return gt_bboxes, gt_labels, gt_ids, gt_ignores, \
            gt_alphas, gt_rotys, gt_dims, gt_trans, gt_2dcs

    def save_pkl(self,
                 img_meta,
                 det_bboxes,
                 det_labels,
                 embeds,
                 det_depths=None,
                 det_dims=None,
                 det_alphas=None,
                 det_2dcs=None,
                 bboxes=None,
                 cls_logits=None,
                 keep_inds=None):
        vid_name = os.path.dirname(
            img_meta[0]['img_info']['file_name']).split('/')[-1]
        img_name = os.path.basename(img_meta[0]['img_info']['file_name'])
        save_path = os.path.join(self.out, 'pkls', vid_name)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, '{}.pkl'.format(img_name))
        to_save = dict(
            det_bboxes=det_bboxes.cpu(),
            det_labels=det_labels.cpu(),
            det_depths=det_depths.cpu() if det_depths is not None else None,
            det_dims=det_dims.cpu() if det_dims is not None else None,
            det_alphas=det_alphas.cpu() if det_alphas is not None else None,
            det_2dcs=det_2dcs.cpu() if det_2dcs is not None else None,
            bboxes=bboxes.cpu() if bboxes else None,
            embeds=embeds.cpu(),
            keep_inds=keep_inds.cpu() if keep_inds else None,
            cls_logits=cls_logits.cpu() if cls_logits else None)
        mmcv.dump(to_save, save_file)

    def show_tracklets(self, img_meta, track_bboxes, track_labels, track_ids):
        vid_name = os.path.dirname(
            img_meta[0]['img_info']['file_name']).split('/')[-1]
        img_name = os.path.basename(img_meta[0]['img_info']['file_name'])
        save_path = os.path.join(self.out, 'shows', vid_name)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, img_name)
        img = os.path.join(self.data.img_prefix, vid_name, img_name)
        img = mmcv.imshow_tracklets(
            img, track_bboxes, track_labels, track_ids, out_file=save_file)

    def save_det_txt(self,
                     outputs,
                     class_cfg,
                     img_meta,
                     use_3d_box_center=False,
                     adjust_center=False):
        """
        #Values    Name      Description
        ----------------------------------------------------------------------
        1   type        Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
        1   truncated   Float from 0 (non-truncated) to 1 (truncated), where
                        truncated refers to the object leaving image boundaries
        1   occluded    Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
        1   alpha       Observation angle of object, ranging [-pi..pi]
        4   bbox        2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
        3   dimensions  3D object dimensions: height, width, length (in meters)
        3   location    3D object location x,y,z in camera coordinates (in meters)
        1   rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1   score       Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.

        Args:
            outputs (dict): prediction results
            class_cfg (dict): a dict to convert class.
            img_meta (dict): image meta information.
        """
        out_folder = os.path.join(self.out, 'txts')
        os.makedirs(out_folder, exist_ok=True)
        img_info = img_meta[0]['img_info']
        txt_file = os.path.join(
            out_folder,
            os.path.splitext(os.path.basename(img_info['file_name']))[0] +
            '.txt')

        # Expand dimension of results
        n_obj_detect = len(outputs['track_results'])
        if outputs.get('depth_results', None) is not None:
            depths = outputs['depth_results'].cpu().numpy().reshape(-1, 1)
        else:
            depths = np.full((n_obj_detect, 1), -1000)
        if outputs.get('dim_results', None) is not None:
            dims = outputs['dim_results'].cpu().numpy().reshape(-1, 3)
        else:
            dims = np.full((n_obj_detect, 3), -1000)
        if outputs.get('alpha_results', None) is not None:
            alphas = outputs['alpha_results'].cpu().numpy().reshape(-1, 1)
        else:
            alphas = np.full((n_obj_detect, 1), -10)
        if outputs.get('cen_2ds_results', None) is not None:
            centers = outputs['cen_2ds_results'].cpu().numpy().reshape(-1, 2)
        else:
            centers = [None] * n_obj_detect

        lines = []
        for (trackId, bbox), depth, dim, alpha, cen in zip(
                outputs['track_results'].items(), depths, dims, alphas,
                centers):
            loc, label = bbox['bbox'], bbox['label']
            if use_3d_box_center and cen is not None:
                box_cen = cen
            else:
                box_cen = np.array([loc[0] + loc[2], loc[1] + loc[3]]) / 2
            if alpha == -10:
                roty = np.full((1, ), -10)
            else:
                roty = tu.alpha2rot_y(alpha,
                                      box_cen[0] - img_info['width'] / 2,
                                      img_info['cali'][0][0])
            if np.all(depths == -1000):
                trans = np.full((3, ), -1000)
            else:
                trans = tu.imagetocamera(box_cen[None], depth,
                                         np.array(img_info['cali'])).flatten()

            if adjust_center:
                # KITTI GT uses the bottom of the car as center (x, 0, z).
                # Prediction uses center of the bbox as center (x, y, z).
                # So we align them to the bottom center as GT does
                trans[1] += dim[0] / 2.0

            if bbox['label'] == class_cfg['Car']:
                cat = 'Car'
            elif bbox['label'] == class_cfg['Pedestrian']:
                cat = 'Pedestrian'
            elif bbox['label'] == class_cfg['Cyclist']:
                cat = 'Cyclist'
            else:
                continue

            # Create lines of results
            line = f"{cat} {-1} {-1} " \
                   f"{alpha.item():.6f} " \
                   f"{loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f} {loc[3]:.6f} " \
                   f"{dim[0]:.6f} {dim[1]:.6f} {dim[2]:.6f} " \
                   f"{trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} " \
                   f"{roty.item():.6f} {loc[4]:.6f}\n"
            lines.append(line)

        if txt_file in self.writed:
            mode = 'a'
        else:
            mode = 'w'
            self.writed.append(txt_file)
        if len(lines) > 0:
            with open(txt_file, mode) as f:
                f.writelines(lines)
        else:
            with open(txt_file, mode):
                pass

    def save_trk_txt(self,
                     outputs,
                     cfg,
                     img_meta,
                     use_3d_box_center=False,
                     adjust_center=False):
        """
        #Values    Name      Description
        ----------------------------------------------------------------------
        1   frame       Frame within the sequence where the object appearers
        1   track id    Unique tracking id of this object within this sequence
        1   type        Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
        1   truncated   Float from 0 (non-truncated) to 1 (truncated), where
                        truncated refers to the object leaving image boundaries.
                        Truncation 2 indicates an ignored object (in particular
                        in the beginning or end of a track) introduced by manual
                        labeling.
        1   occluded    Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
        1   alpha       Observation angle of object, ranging [-pi..pi]
        4   bbox        2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
        3   dimensions  3D object dimensions: height, width, length (in meters)
        3   location    3D object location x,y,z in camera coordinates (in meters)
        1   rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1   score       Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.

        Args:
            outputs (dict): prediction results
            class_cfg (dict): a dict to convert class.
            img_meta (dict): image meta information.
        """
        out_folder = os.path.join(self.out, 'txts')
        os.makedirs(out_folder, exist_ok=True)
        img_info = img_meta[0]['img_info']
        vid_name = os.path.dirname(img_info['file_name']).split('/')[-1]
        txt_file = os.path.join(out_folder, '{}.txt'.format(vid_name))

        # Expand dimension of results
        n_obj_detect = len(outputs['track_results'])
        if outputs.get('depth_results', None) is not None:
            depths = outputs['depth_results'].cpu().numpy().reshape(-1, 1)
        else:
            depths = np.full((n_obj_detect, 1), -1000)
        if outputs.get('dim_results', None) is not None:
            dims = outputs['dim_results'].cpu().numpy().reshape(-1, 3)
        else:
            dims = np.full((n_obj_detect, 3), -1000)
        if outputs.get('alpha_results', None) is not None:
            alphas = outputs['alpha_results'].cpu().numpy().reshape(-1, 1)
        else:
            alphas = np.full((n_obj_detect, 1), -10)

        if outputs.get('cen_2ds_results', None) is not None:
            centers = outputs['cen_2ds_results'].cpu().numpy().reshape(-1, 2)
        else:
            centers = [None] * n_obj_detect

        lines = []
        for (trackId, bbox), depth, dim, alpha, cen in zip(
                outputs['track_results'].items(), depths, dims, alphas,
                centers):
            loc, label = bbox['bbox'], bbox['label']
            if use_3d_box_center and cen is not None:
                box_cen = cen
            else:
                box_cen = np.array([loc[0] + loc[2], loc[1] + loc[3]]) / 2
            if alpha == -10:
                roty = np.full((1, ), -10)
            else:
                roty = tu.alpha2rot_y(alpha,
                                      box_cen[0] - img_info['width'] / 2,
                                      img_info['cali'][0][0])
            if np.all(depths == -1000):
                trans = np.full((3, ), -1000)
            else:
                trans = tu.imagetocamera(box_cen[None], depth,
                                         np.array(img_info['cali'])).flatten()

            if adjust_center:
                # KITTI GT uses the bottom of the car as center (x, 0, z).
                # Prediction uses center of the bbox as center (x, y, z).
                # So we align them to the bottom center as GT does
                trans[1] += dim[0] / 2.0

            cat = ''
            for key in cfg:
                if bbox['label'] == cfg[key]:
                    cat = key.lower()
                    break
            
            if cat == '':
                continue

            # Create lines of results
            line = f"{img_info['index']} {trackId} {cat} {-1} {-1} " \
                   f"{alpha.item():.6f} " \
                   f"{loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f} {loc[3]:.6f} " \
                   f"{dim[0]:.6f} {dim[1]:.6f} {dim[2]:.6f} " \
                   f"{trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} " \
                   f"{roty.item():.6f} {loc[4]:.6f}\n"
            lines.append(line)

        if txt_file in self.writed:
            mode = 'a'
        else:
            mode = 'w'
            self.writed.append(txt_file)
        if len(lines) > 0:
            with open(txt_file, mode) as f:
                f.writelines(lines)
        else:
            with open(txt_file, mode):
                pass

    def plt_tracklets(self, img_meta, track_bboxes, track_labels, track_ids):
        vid_name = os.path.dirname(
            img_meta[0]['img_info']['file_name']).split('/')[-1]
        img_name = os.path.basename(img_meta[0]['img_info']['file_name'])
        save_path = os.path.join(self.out, 'shows', vid_name)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, img_name.split('-')[-1])
        img_name = os.path.join(self.data.img_prefix, vid_name, img_name)

        img = imshow_bboxes_w_ids(
            img_name, track_bboxes, track_ids, out_file=save_file)

    def plt_3d_tracklets(self, img_meta, track_bboxes, track_labels,
                         track_depths, track_dims, track_alphas, track_2dcs,
                         track_ids):
        vid_name = os.path.dirname(
            img_meta[0]['img_info']['file_name']).split('/')[-1]
        img_name = os.path.basename(img_meta[0]['img_info']['file_name'])
        save_path = os.path.join(self.out, 'shows', vid_name)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, img_name.split('-')[-1])
        # img_name = os.path.join(
        #     self.data.img_prefix[img_meta[0]['img_info']['type']], vid_name,
        #     img_name)
        img_name = img_meta[0]['img_info']['file_name']

        img = imshow_3d_tracklets(
            img_name,
            track_bboxes,
            track_labels,
            ids=track_ids,
            depths=track_depths,
            out_file=save_file)
