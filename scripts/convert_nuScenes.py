import os
import os.path as osp
import copy
from tqdm import tqdm
from typing import Tuple, Union
from pathlib import Path
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import view_points, transform_matrix, BoxVisibility
from nuscenes.utils.data_classes import Box

from scripts.bdd_utils import dump_json

trackid_maps = dict()

cats_mapping = {
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

cls_range = {
    'bicycle': 40,
    'motorcycle': 40,
    'pedestrian': 40,
    'bus': 50,
    'car': 50,
    'trailer': 50,
    'truck': 50,
    'construction_vehicle': 50,
    'traffic_cone': 30,
    'barrier': 30
}

USED_SENSOR = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
    'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]

SENSOR_ID = {
    'CAM_FRONT': 1,
    'CAM_FRONT_RIGHT': 2,
    'CAM_BACK_RIGHT': 3,
    'CAM_BACK': 4,
    'CAM_BACK_LEFT': 5,
    'CAM_FRONT_LEFT': 6
}

seq_map = {
    "train_seq_id": 0,
    "val_seq_id": 700,
    "mini_train_seq_id": 850,
    "mini_val_seq_id": 858,
    "test_seq_id": 860
}

general_to_nusc_cats = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec["next"] == "":
                sd_rec = nusc.get("sample_data", sd_rec["next"])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)

    print("exist scene num:", len(available_scenes))

    return available_scenes


def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2Î
    pts_3d_homo = np.concatenate(
        [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d


def project_kitti_box_to_image(box: Box, p_left: np.ndarray, imsize: Tuple[int, int]) \
        -> Union[None, Tuple[int, int, int, int]]:
    """
    Projects 3D box into image FOV.
    :param box: 3D box in camera coordinate
    :param p_left: <np.float: 3, 4>. Projection matrix.
    :param imsize: (width, height). Image size.
    :return: (xmin, ymin, xmax, ymax). Bounding box in image plane or None if box is not in the image.
    """

    # Check that some corners are inside the image.
    corners = np.array([corner for corner in box.corners().T
                        if corner[2] > 0]).T
    if len(corners) == 0:
        return None

    # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
    imcorners = view_points(corners, p_left, normalize=True)[:2]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]),
            np.max(imcorners[1]))

    # Crop bbox to prevent it extending outside image.
    bbox_crop = tuple(max(0, b) for b in bbox)
    bbox_crop = (min(imsize[0], bbox_crop[0]), min(imsize[0], bbox_crop[1]),
                 min(imsize[0], bbox_crop[2]), min(imsize[1], bbox_crop[3]))

    # Detect if a cropped box is empty.
    if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
        return None

    return bbox_crop


def _bbox_inside(box1, box2):
    return box1[0] > box2[0] and box1[0] + box1[2] < box2[0] + box2[2] and \
            box1[1] > box2[1] and box1[1] + box1[3] < box2[1] + box2[3]


def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha


def nusc2coco(data_dir, nusc, scenes, input_seq_id):

    ret = defaultdict(list)

    for k, v in cats_mapping.items():
        ret['categories'].append(dict(id=v, name=k))

    vid_id = 0
    fr_id = 0

    for sensor in USED_SENSOR:
        print(f'converting {sensor}')
        seq_id = input_seq_id

        for sample in tqdm(nusc.sample):
            if not (sample["scene_token"] in scenes):
                continue

            image_token = sample['data'][sensor]

            sd_record = nusc.get('sample_data', image_token)
            img_name = osp.join(data_dir, sd_record['filename'])
            height = sd_record['height']
            width = sd_record['width']

            cs_record = nusc.get('calibrated_sensor',
                                 sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
            global_from_car = transform_matrix(pose_record['translation'],
                                               Quaternion(
                                                   pose_record['rotation']),
                                               inverse=False)
            car_from_sensor = transform_matrix(cs_record['translation'],
                                               Quaternion(
                                                   cs_record['rotation']),
                                               inverse=False)
            trans_matrix = np.dot(global_from_car, car_from_sensor)

            rotation_matrix = trans_matrix[:3, :3]
            position = trans_matrix[:3, 3:].flatten().tolist()

            rotation = R.from_matrix(rotation_matrix).as_euler('xyz').tolist()
            pose_dict = dict(rotation=rotation, position=position)

            _, boxes, camera_intrinsic = nusc.get_sample_data(
                image_token, box_vis_level=BoxVisibility.ANY)
            calib = np.eye(4, dtype=np.float32)
            calib[:3, :3] = camera_intrinsic
            calib = calib[:3].tolist()

            img_info = dict(file_name=img_name,
                            cali=calib,
                            pose=pose_dict,
                            sensor_id=SENSOR_ID[sensor],
                            height=height,
                            width=width,
                            fov=60,
                            near_clip=0.15,
                            id=len(ret['images']),
                            video_id=vid_id,
                            index=fr_id)
            ret['images'].append(img_info)

            anns = []
            for box in boxes:
                cat = general_to_nusc_cats[box.name]
                if cat in ['ignore']:
                    continue

                v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
                yaw = -np.arctan2(v[2], v[0]).tolist()

                center_2d = project_to_image(
                    np.array([box.center[0], box.center[1], box.center[2]],
                             np.float32).reshape(1, 3), calib)[0].tolist()

                sample_ann = nusc.get('sample_annotation', box.token)
                instance_token = sample_ann['instance_token']

                if instance_token not in trackid_maps.keys():
                    trackid_maps[instance_token] = len(trackid_maps)

                bbox = project_kitti_box_to_image(copy.deepcopy(box),
                                                  camera_intrinsic,
                                                  imsize=(width, height))

                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                alpha = _rot_y2alpha(yaw, (x1 + x2) / 2, camera_intrinsic[0,
                                                                          2],
                                     camera_intrinsic[0, 0]).tolist()
                delta_2d = [
                    center_2d[0] - (x1 + x2) / 2, center_2d[1] - (y1 + y2) / 2
                ]
                ann = dict(
                    image_id=ret['images'][-1]['id'],
                    category_id=cats_mapping[cat],
                    instance_id=trackid_maps[instance_token],
                    alpha=float(alpha),
                    roty=float(yaw),
                    dimension=[box.wlh[2], box.wlh[0], box.wlh[1]],
                    translation=[box.center[0], box.center[1], box.center[2]],
                    is_occluded=0,
                    is_truncated=0,
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    area=(x2 - x1) * (y2 - y1),
                    delta_2d=delta_2d,
                    center_2d=center_2d,
                    iscrowd=False,
                    ignore=False,
                    segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]])
                anns.append(ann)

            # Filter out bounding boxes outside the image
            visable_anns = []
            for i in range(len(anns)):
                vis = True
                for j in range(len(anns)):
                    if anns[i]['translation'][2] - min(anns[i]['dimension']) / 2 > \
                        anns[j]['translation'][2] + max(anns[j]['dimension']) / 2 and \
                        _bbox_inside(anns[i]['bbox'], anns[j]['bbox']):
                        vis = False
                        break
                if vis:
                    visable_anns.append(anns[i])
                else:
                    pass

            for ann in visable_anns:
                ann['id'] = len(ret['annotations'])
                ret['annotations'].append(ann)

            fr_id += 1
            if sample['next'] == '':
                vid_info = dict(id=vid_id,
                                name=f'{seq_id:05}_{SENSOR_ID[sensor]}',
                                scene_token=sample["scene_token"],
                                n_frames=fr_id)
                ret['videos'].append(vid_info)
                seq_id += 1
                vid_id += 1
                fr_id = 0

    return ret


def convert_track(data_dir, version):

    if not osp.exists(osp.join(data_dir, version)):
        print(f"Folder {osp.join(data_dir, version)} is not found")
        if "test" in version:
            return None
        else:
            return None, None

    nusc = NuScenes(version=version, dataroot=data_dir, verbose=True)

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers

    if version == "v1.0-trainval":
        train_scenes = splits.train
        train_seq_id = seq_map['train_seq_id']
        val_scenes = splits.val
        val_seq_id = seq_map['val_seq_id']
    elif version == "v1.0-test":
        train_scenes = splits.test
        train_seq_id = seq_map['test_seq_id']
        val_scenes = []
        val_seq_id = 0
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        train_seq_id = seq_map['mini_train_seq_id']
        val_scenes = splits.mini_val
        val_seq_id = seq_map['mini_val_seq_id']
    else:
        raise ValueError("unknown")

    test = "test" in version

    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]

    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])

    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    if test:
        trainset = 'testing'
    else:
        trainset = 'training'

    print('=====')
    print(f'Converting {trainset} set')
    print('=====')
    train_anns = nusc2coco(data_dir, nusc, train_scenes, train_seq_id)

    if test:
        return train_anns
    else:
        print('=====')
        print('Converting validation set')
        print('=====')
        val_anns = nusc2coco(data_dir, nusc, val_scenes, val_seq_id)
        return train_anns, val_anns


def main():
    data_dir = 'data/nuscenes'
    out_dir = 'data/nuscenes/anns'

    print('Convert Nuscenes Tracking dataset to COCO style.')
    if not osp.isfile(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("tracking mini")
    train_anns, val_anns = convert_track(data_dir, version='v1.0-mini')
    dump_json(osp.join(out_dir, 'tracking_train_mini.json'), train_anns)
    dump_json(osp.join(out_dir, 'tracking_val_mini.json'), val_anns)

    print("tracking trainval")
    train_anns, val_anns = convert_track(data_dir, version='v1.0-trainval')
    dump_json(osp.join(out_dir, 'tracking_train.json'), train_anns)
    dump_json(osp.join(out_dir, 'tracking_val.json'), val_anns)

    print('tracking test')
    anns = convert_track(data_dir, version='v1.0-test')
    dump_json(osp.join(out_dir, 'tracking_test.json'), anns)


if __name__ == "__main__":
    main()