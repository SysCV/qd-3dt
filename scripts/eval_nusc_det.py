import os
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

model_cats = dict(bicycle=0, motorcycle=1, pedestrian=2, bus=3, car=4,
                  trailer=5, truck=6, construction_vehicle=7, traffic_cone=8, barrier=9)

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}

seq_map = {
    "train_seq_id": 0,
    "val_seq_id": 700,
    "mini_train_seq_id": 850,
    "mini_val_seq_id": 858,
    "test_seq_id": 860
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nuscenes Detection Evaluation")
    parser.add_argument("--version", help="Nuscenes dataset version")
    parser.add_argument("--root", help="Nuscenes dataset root")
    parser.add_argument(
        "--work_dir", help="the dir which saves the tracking results json file")
    parser.add_argument("--gt_anns", help="gt json file")

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

    print("Begin Converting Tracking Result")
    for i in tqdm(range(len(tracking_result['videos']))):
        prev_loc = dict()
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

        for sample_id in range(scene['nbr_samples']):
            annos = []
            img_id = first_frame_id + sample_id
            pose_dict = tracking_result['images'][img_id]['pose']
            projection = tracking_result['images'][img_id]['cali']

            cam_to_global = R.from_euler(
                'xyz', pose_dict['rotation']).as_matrix()

            outputs = [ann for ann in tracking_result['annotations'] if ann['image_id'] == img_id]

            for output in outputs:
                # filter ignore objects
                name = cats_mapping[output['category_id']]
                if name == 'ignore':
                    continue

                # move to world coordinate
                translation = np.dot(
                    cam_to_global, np.array(output['translation']))
                translation += np.array(pose_dict['position'])

                quat = Quaternion(axis=[0, 1, 0], radians=output['roty'])
                x, y, z, w = R.from_matrix(cam_to_global).as_quat()
                rotation = Quaternion([w, x, y, z]) * quat

                h, w, l = output['dimension']
                dimension = [w, l, h]
                for i in range(len(dimension)):
                    if dimension[i] <= 0:
                        dimension[i] = 0.1

                tracking_id = str(output['instance_id'])
                if prev_loc.get(tracking_id) is not None:
                    velocity = translation - prev_loc[tracking_id]
                else:
                    velocity = [0.0, 0.0, 0.0]
                prev_loc[tracking_id] = translation

                nusc_anno = {
                    "sample_token": token,
                    "translation": translation.tolist(),
                    "size": dimension,
                    "rotation": rotation.elements.tolist(),
                    "velocity": [velocity[0], velocity[1]],
                    "detection_name": name,
                    "detection_score": output['score'] * output['uncertainty'],
                    "attribute_name": max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }

                # align six camera
                if nusc_annos["results"].get(token) is not None:
                    nms_flag = 0
                    for all_cam_ann in nusc_annos["results"][token]:
                        if nusc_anno['detection_name'] == all_cam_ann['detection_name']:
                            translation = nusc_anno['translation']
                            ref_translation = all_cam_ann['translation']
                            translation_diff = (translation[0] - ref_translation[0],
                                                translation[1] - ref_translation[1],
                                                translation[2] - ref_translation[2])
                            if nusc_anno['detection_name'] in ['pedestrian']:
                                nms_dist = 1
                            else:
                                nms_dist = 2
                            if np.sqrt(np.sum(np.array(translation_diff[:2]) ** 2)) < nms_dist:
                                if all_cam_ann['detection_score'] < nusc_anno['detection_score']:
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

    with open(os.path.join(args.work_dir, 'detection_result.json'), "w") as f:
        json.dump(nusc_annos, f)

    return nusc


def eval_detection(args, nusc):
    if 'mini' in args.version:
        eval_set = 'mini_val'
    else:
        eval_set = 'val'

    eval(os.path.join(args.work_dir, 'detection_result.json'),
         eval_set,
         args.work_dir,
         args.root,
         nusc
         )


def eval(res_path, eval_set, output_dir=None, root_path=None, nusc=None):
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.detection.config import config_factory

    cfg = config_factory("detection_cvpr_2019")

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(render_curves=False)


def main():
    args = parse_args()

    nusc = convert_track(args)

    if args.version != 'v1.0-test':
        eval_detection(args, nusc)


if __name__ == '__main__':
    main()
