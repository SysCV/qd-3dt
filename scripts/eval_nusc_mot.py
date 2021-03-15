import os
import subprocess
import sys
import json
import numpy as np
import copy
import argparse
import operator
from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix

USED_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

SENSOR_ID = {
    'CAM_FRONT': 1,
    'CAM_FRONT_RIGHT': 2,
    'CAM_BACK_RIGHT': 3,
    'CAM_BACK': 4,
    'CAM_BACK_LEFT': 5,
    'CAM_FRONT_LEFT': 6
}

cats_mapping = {
    1: 'bicycle',
    2: 'motorcycle',
    3: 'pedestrian',
    4: 'bus',
    5: 'car',
    6: 'trailer',
    7: 'truck',
    8: 'construction_vehicle',
    9: 'traffic_cone',
    10: 'barrier',
    11: 'ignore'
}

model_cats = dict(bicycle=0, motorcycle=1, pedestrian=2,
                  bus=3, car=4, trailer=5, truck=6)

seq_map = {
    "train_seq_id": 0,
    "val_seq_id": 700,
    "mini_train_seq_id": 850,
    "mini_val_seq_id": 858,
    "test_seq_id": 860
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nuscenes Tracking Evaluation")
    parser.add_argument("--version", help="Nuscenes dataset version")
    parser.add_argument("--root", help="Nuscenes dataset root")
    parser.add_argument(
        "--work_dir", help="the dir which saves the tracking results json file")
    parser.add_argument("--gt_anns", help="gt json file")
    parser.add_argument("--amota_02", "--amota_02", action='store_true', 
                        help="run AMOTA@0.2")
    args = parser.parse_args()

    return args


def convert_track(args):

    tracking_result_path = os.path.join(args.work_dir, 'output.json')

    print(f'Loading tracking result from {tracking_result_path}')

    with open(tracking_result_path, 'rb') as f:
        tracking_result = json.load(f)

    with open(args.gt_anns, 'rb') as f:
        gt = json.load(f)

    nusc = NuScenes(version=f'{args.version}',
                    dataroot=f'{args.root}', verbose=True)

    if args.version == 'v1.0-mini':
        start_seq_id = seq_map['mini_val_seq_id']
        vid_num = 2
    elif args.version == 'v1.0-trainval':
        start_seq_id = seq_map['val_seq_id']
        vid_num = 150
    else:
        start_seq_id = seq_map['test_seq_id']
        vid_num = 150

    for sensor in USED_SENSOR:
        # grep scene token
        for vid_info in tracking_result['videos']:
            for gt_vid_info in gt['videos']:
                if f"{int(vid_info['id']) - vid_num*(SENSOR_ID[sensor]-1) + start_seq_id:05}_{SENSOR_ID[sensor]}" == gt_vid_info['name']:
                    vid_info['scene_token'] = gt_vid_info['scene_token']
                    vid_info['sensor'] = sensor

    del gt

    nusc_annos = {
        "results": {},
        "meta": None,
    }

    first_frame_id = -1
    memo = []
    first_frame_token = []

    print("Begin Converting Tracking Result")
    for i in tqdm(range(len(tracking_result['videos']))):

        sensor = tracking_result['videos'][i]['sensor']
        scene = nusc.get('scene', tracking_result['videos'][i]['scene_token'])

        # find start frame for each video
        for frame in tracking_result['images']:
            if frame['first_frame'] and not frame['id'] in memo:
                first_frame_id = frame['id']
                memo.append(frame['id'])
                break

        # First sample (key frame)
        token = scene['first_sample_token']
        sample = nusc.get('sample', token)

        first_frame_token.append(token)

        for sample_id in range(scene['nbr_samples']):
            annos = []
            img_id = first_frame_id + sample_id
            pose_dict = tracking_result['images'][img_id]['pose']
            projection = tracking_result['images'][img_id]['cali']

            cam_to_global = R.from_euler(
                'xyz', pose_dict['rotation']).as_matrix()

            outputs = [ann for ann in tracking_result['annotations'] if ann['image_id'] == img_id]

            for output in outputs:
                # remove non-tracking objects
                cat = ''
                for key in model_cats:
                    if cats_mapping[output['category_id']] == key:
                        cat = key.lower()
                        break

                if cat == '':
                    continue

                # move to world coordinate
                translation = np.dot(
                    cam_to_global, np.array(output['translation']))
                translation += np.array(pose_dict['position'])

                quat = Quaternion(axis=[0, 1, 0], radians=output['roty'])
                x, y, z, w = R.from_matrix(cam_to_global).as_quat()
                rotation = Quaternion([w, x, y, z]) * quat

                tracking_id = str(output['instance_id'])

                h, w, l = output['dimension']

                nusc_anno = {
                    "sample_token": token,
                    "translation": translation.tolist(),
                    "size": [w, l, h],
                    "rotation": rotation.elements.tolist(),
                    "velocity": [0., 0.],
                    "tracking_id": tracking_id,
                    "tracking_name": cat,
                    "tracking_score": output['score'] * output['uncertainty'],
                }

                # align six camera
                if nusc_annos["results"].get(token) is not None:
                    nms_flag = 0
                    for all_cam_ann in nusc_annos["results"][token]:
                        if nusc_anno['tracking_name'] == all_cam_ann['tracking_name']:
                            translation = nusc_anno['translation']
                            ref_translation = all_cam_ann['translation']
                            translation_diff = (translation[0] - ref_translation[0],
                                                translation[1] - ref_translation[1],
                                                translation[2] - ref_translation[2])
                            if nusc_anno['tracking_name'] in ['pedestrian']:
                                nms_dist = 1
                            else:
                                nms_dist = 2
                            if np.sqrt(np.sum(np.array(translation_diff[:2]) ** 2)) < nms_dist:
                                if all_cam_ann['tracking_score'] < nusc_anno['tracking_score']:
                                    all_cam_ann = nusc_anno
                                nms_flag = 1
                                break
                    if nms_flag == 0:
                        annos.append(nusc_anno)
                else:
                    annos.append(nusc_anno)

            if nusc_annos["results"].get(token) is not None:
                annos += nusc_annos["results"][token]
            nusc_annos["results"].update({token: annos})

            token = sample['next']
            if token != '':
                sample = nusc.get('sample', token)

    nusc_annos["meta"] = {
        "use_camera": True,
        "use_lidar": False,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)

    return nusc


def eval_tracking(args):
    if 'mini' in args.version:
        eval_set = 'mini_val'
    else:
        eval_set = 'val'

    eval(os.path.join(args.work_dir, 'tracking_result.json'),
         eval_set,
         args.version,
         args.work_dir,
         args.root
         )


def eval(res_path, eval_set, version, output_dir=None, root_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval as track_eval
    from nuscenes.eval.common.config import config_factory as track_configs

    cfg = track_configs("tracking_nips_2019")

    nusc_eval = track_eval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version=version,
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()


def main():
    args = parse_args()

    if not args.amota_02:
        convert_track(args)

    if args.version != 'v1.0-test':
        eval_tracking(args)


if __name__ == '__main__':
    main()
