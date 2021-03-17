import os
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix

from scripts.bdd_utils import dump_json

USED_SENSOR = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
    'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]

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


def nusc2coco(data_dir, nusc, scenes, input_seq_id):
    ret = defaultdict(list)

    for k, v in cats_mapping.items():
        ret['categories'].append(dict(id=v, name=k))

    ret['annotations'] = []

    vid_id = 0
    fr_id = 0
    key_fr_id = 0

    for sensor in USED_SENSOR:
        print(f'converting {sensor}')
        seq_id = input_seq_id

        first_samples = [
            sample for sample in nusc.sample
            if sample['prev'] == '' and sample["scene_token"] in scenes
        ]

        for sample in tqdm(first_samples):

            sensor_data = nusc.get('sample_data', sample['data'][sensor])

            while 1:
                img_name = osp.join(data_dir, sensor_data['filename'])
                height = sensor_data['height']
                width = sensor_data['width']

                cs_record = nusc.get('calibrated_sensor',
                                     sensor_data['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose',
                                       sensor_data['ego_pose_token'])
                global_from_car = transform_matrix(
                    pose_record['translation'],
                    Quaternion(pose_record['rotation']),
                    inverse=False)
                car_from_sensor = transform_matrix(cs_record['translation'],
                                                   Quaternion(
                                                       cs_record['rotation']),
                                                   inverse=False)
                trans_matrix = np.dot(global_from_car, car_from_sensor)

                rotation_matrix = trans_matrix[:3, :3]
                position = trans_matrix[:3, 3:].flatten().tolist()

                rotation = R.from_matrix(rotation_matrix).as_euler(
                    'xyz').tolist()
                pose_dict = dict(rotation=rotation, position=position)

                camera_intrinsic = cs_record['camera_intrinsic']
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
                                index=fr_id,
                                key_frame_index=key_fr_id,
                                is_key_frame=sensor_data['is_key_frame'])
                ret['images'].append(img_info)

                if sensor_data['is_key_frame']:
                    key_fr_id += 1

                fr_id += 1
                if sensor_data['next'] != '':
                    sensor_data = nusc.get('sample_data', sensor_data['next'])
                else:
                    vid_info = dict(id=vid_id,
                                    name=f'{seq_id:05}_{SENSOR_ID[sensor]}',
                                    scene_token=sample["scene_token"],
                                    n_frames=fr_id)
                    ret['videos'].append(vid_info)
                    seq_id += 1
                    vid_id += 1
                    fr_id = 0
                    key_fr_id = 0
                    break

    return ret


def convert_track(data_dir, version):
    if not osp.exists(osp.join(data_dir, version)):
        print(f"Folder {osp.join(data_dir, version)} is not found")
        return None
    
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
        print(f"val scene: {len(val_scenes)}")

    print('=====')
    if test:
        print('Converting testing set')
        val_anns = nusc2coco(data_dir, nusc, train_scenes, val_seq_id)
    else:
        print('Converting validation set')
        val_anns = nusc2coco(data_dir, nusc, val_scenes, val_seq_id)
    print('=====')

    return val_anns


def main():
    data_dir = 'data/nuscenes'
    out_dir = 'data/nuscenes/anns'

    print('Convert Nuscenes Tracking dataset (full frames) to COCO style.')
    if not osp.isfile(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("tracking mini")
    anns = convert_track(data_dir, version='v1.0-mini')
    dump_json(osp.join(out_dir, 'tracking_val_mini_full_frames.json'), anns)

    print("tracking trainval")
    anns = convert_track(data_dir, version='v1.0-trainval')
    dump_json(osp.join(out_dir, 'tracking_val_full_frames.json'), anns)

    print('tracking test')
    anns = convert_track(data_dir, version='v1.0-test')
    dump_json(osp.join(out_dir, 'tracking_test_full_frames.json'), anns)


if __name__ == "__main__":
    main()
