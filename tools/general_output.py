import os.path as osp
import numpy as np

import scripts.tracking_utils as tu

kitti_gta_mapping = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    'truck': 4,
    'tram': 5,
    'misc': 6,
    'dontcare': 7
}

nusc_mapping = {
    'bicycle': 1,
    'motorcycle': 2,
    'pedestrian': 3,
    'bus': 4,
    'car': 5,
    'trailer': 6,
    'truck': 7,
    'construction_vehicle': 8,
    'traffic_cone': 9,
    'barrier': 10,
    'ignore': 11
}

waymo_mapping = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    'dontcare': 4
}


def general_output(coco_json, outputs, img_info, use_3d_box_center, pred_id,
                   modelcats, out_path):
    if 'Nusc' in out_path:
        cats_mapping = nusc_mapping
    elif 'Waymo' in out_path:
        cats_mapping = waymo_mapping
    else:
        cats_mapping = kitti_gta_mapping

    if not ('categories' in coco_json.keys()):
        for k, v in cats_mapping.items():
            coco_json['categories'].append(dict(id=v, name=k))

    if img_info.get(
        'is_key_frame') is not None and img_info['is_key_frame']:
        img_info['index'] = img_info['key_frame_index']

    img_info['id'] = len(coco_json['images'])
    vid_name = osp.dirname(img_info['file_name']).split('/')[-1]
    if img_info['first_frame']:
        coco_json['videos'].append(
            dict(id=img_info['video_id'], name=vid_name))

    # pruning img_info
    img_info.pop('filename')
    img_info.pop('type')
    coco_json['images'].append(img_info)

    # Expand dimension of results
    n_obj_detect = len(outputs['track_results'])
    if outputs.get('depth_results', None) is not None:
        depths = outputs['depth_results'].cpu().numpy().reshape(-1, 1)
    else:
        depths = np.ones([n_obj_detect, 1]) * -1000
    if outputs.get('dim_results', None) is not None:
        dims = outputs['dim_results'].cpu().numpy().reshape(-1, 3)
    else:
        dims = np.ones([n_obj_detect, 3]) * -1000
    if outputs.get('alpha_results', None) is not None:
        alphas = outputs['alpha_results'].cpu().numpy().reshape(-1, 1)
    else:
        alphas = np.ones([n_obj_detect, 1]) * -10
    if outputs.get('cen_2ds_results', None) is not None:
        centers = outputs['cen_2ds_results'].cpu().numpy().reshape(-1, 2)
    else:
        centers = [None] * n_obj_detect
    if outputs.get('depth_uncertainty_results', None) is not None:
        depths_uncertainty = outputs['depth_uncertainty_results'].cpu().numpy().reshape(-1, 1)
    else:
        depths_uncertainty = [None] * n_obj_detect

    for (trackId,
         bbox), depth, dim, alpha, cen, depth_uncertainty, in zip(outputs['track_results'].items(),
                                    depths, dims, alphas, centers, depths_uncertainty):
        box = bbox['bbox'].astype(float).tolist()
        cat = ''

        for key in modelcats:
            if bbox['label'] == modelcats[key]:
                cat = key.lower()
                break

        if cat == '':
            continue

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        score = box[4]
        if use_3d_box_center and cen is not None:
            box_cen = cen
        else:
            box_cen = np.array([x1 + x2, y1 + y2]) / 2
        if alpha == -10:
            rot_y = -10
        else:
            rot_y = tu.alpha2rot_y(alpha, box_cen[0] - img_info['width'] / 2,
                                   img_info['cali'][0][0])
        if np.all(depths == -1000):
            trans = np.ones([1, 3]) * -1000
        else:
            trans = tu.imagetocamera(box_cen[np.newaxis], depth,
                                     np.array(img_info['cali'])).flatten()
        ann = dict(
            id=pred_id,
            image_id=img_info['id'],
            category_id=cats_mapping[cat],
            instance_id=trackId.tolist(),
            alpha=float(alpha),
            roty=float(rot_y),
            dimension=dim.astype(float).tolist(),
            translation=trans.astype(float).tolist(),
            is_occluded=False,
            is_truncated=False,
            bbox=[x1, y1, x2 - x1, y2 - y1],
            area=(x2 - x1) * (y2 - y1),
            center_2d=box_cen.astype(float).tolist(),
            uncertainty=float(depth_uncertainty),
            depth=depth.tolist(),
            iscrowd=False,
            ignore=False,
            segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
            score=score)
        coco_json['annotations'].append(ann)
        pred_id += 1

    return coco_json, pred_id
