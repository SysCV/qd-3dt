# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/

import os
import os.path as osp
import argparse
import sys
import json
import subprocess
from tqdm import tqdm

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

camera_dict = {
    'front': dataset_pb2.CameraName.FRONT,
    'front_left': dataset_pb2.CameraName.FRONT_LEFT,
    'front_right': dataset_pb2.CameraName.FRONT_RIGHT,
    'side_left': dataset_pb2.CameraName.SIDE_LEFT,
    'side_right': dataset_pb2.CameraName.SIDE_RIGHT
}

obj_type_dict = {
    1: label_pb2.Label.TYPE_PEDESTRIAN,
    2: label_pb2.Label.TYPE_CYCLIST,
    3: label_pb2.Label.TYPE_VEHICLE
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Waymo 3D Detection Evaluation")
    parser.add_argument("--phase", help="Waymo dataset phase")
    parser.add_argument(
        "--work_dir", help="the dir which saves the tracking results json file")
    parser.add_argument("--gt_bin", help="gt.bin file")

    args = parser.parse_args()

    return args


def convert_track(tracking_result_path):
    print(f'Loading tracking result from {tracking_result_path}')

    with open(tracking_result_path, 'r') as f:
        tracking_result = json.load(f)

    waymo_annos = dict()

    print("Begin Converting Tracking Result")
    for img_info in tqdm(tracking_result['images']):
        file_name = img_info['file_name']
        camera, subset, vid_name, img_name = file_name.split('/')[-4:]
        frame_ind = int(img_info['index'])
        frame_timestamp_micros = img_info['timestamp']

        car_from_cam = np.array(img_info['car_from_cam'])

        outputs = [ann for ann in tracking_result['annotations'] if ann['image_id'] == img_info['id']]

        annos = []
        token = f"{vid_name}_{frame_ind}"

        for output in outputs:
            # remove ignored objects
            if output['category_id'] == 4:
                continue

            cat = obj_type_dict[output['category_id']]

            # move to vehicle frame
            rotation_matrix = car_from_cam[:3, :3]

            x, y, z = output['translation']
            translation = np.matmul(car_from_cam,
                                np.array([x, y, z, 1]).T).tolist()[:3]

            quat = Quaternion(axis=[0, 1, 0], radians=output['roty']).elements
            x, y, z, w = R.from_matrix(rotation_matrix).as_quat()
            w, x, y, z = Quaternion([w, x, y, z]) * quat

            heading = R.from_quat([x, y, z, w]).as_euler('xyz')[2]

            score = output['score'] * output['uncertainty']

            for i in range(len(output['dimension'])):
                if output['dimension'][i] <= 0:
                    output['dimension'][i] = 0.1

            waymo_anno = {
                "context_name": vid_name,
                "frame_timestamp_micros": frame_timestamp_micros,
                "ego_box_3d": translation,
                "dimension": output['dimension'],
                "ego_heading": heading,
                "score": score,
                "object_id": output['instance_id'],
                "cat": cat
            }

            # align five camera
            if waymo_annos.get(token) is not None:
                nms_flag = 0
                for all_cam_ann in waymo_annos[token]:
                    if waymo_anno['cat'] == all_cam_ann['cat']:
                        translation = waymo_anno['ego_box_3d']
                        ref_translation = all_cam_ann['ego_box_3d']
                        translation_diff = (translation[0] - ref_translation[0],
                                            translation[1] - ref_translation[1],
                                            translation[2] - ref_translation[2])
                        if waymo_anno['cat'] in [1]:
                            nms_dist = 1
                        else:
                            nms_dist = 2
                        if np.sqrt(np.sum(np.array(translation_diff[:2]) ** 2)) < nms_dist:
                            if all_cam_ann['score'] < waymo_anno['score']:
                                all_cam_ann = waymo_anno
                            nms_flag = 1
                            break
                if nms_flag == 0:
                    annos.append(waymo_anno)
            else:
                annos.append(waymo_anno)

        if waymo_annos.get(token) is not None:
                annos += waymo_annos[token]
        waymo_annos.update({token: annos})

    return waymo_annos


def create_waymo_sumbit(waymo_annos, out_path):
    print('Creating Waymo submission...')
    objects = metrics_pb2.Objects()

    for token in tqdm(waymo_annos):
        for waymo_anno in waymo_annos[token]:
            o = metrics_pb2.Object()
            o.context_name = waymo_anno['context_name']
            o.frame_timestamp_micros = waymo_anno['frame_timestamp_micros']

            box = label_pb2.Label.Box()
            box.center_x = waymo_anno['ego_box_3d'][0]
            box.center_y = waymo_anno['ego_box_3d'][1]
            box.center_z = waymo_anno['ego_box_3d'][2]
            box.length = waymo_anno['dimension'][2]
            box.width = waymo_anno['dimension'][1]
            box.height = waymo_anno['dimension'][0]
            box.heading = waymo_anno['ego_heading']
            o.object.box.CopyFrom(box)

            o.score = waymo_anno['score']
            o.object.id = str(waymo_anno['object_id'])
            o.object.type = waymo_anno['cat']

            o.object.num_lidar_points_in_box = 100

            objects.objects.append(o)

    # Write objects to a file.
    with open(out_path, 'wb') as f:
        f.write(objects.SerializeToString())


def eval_detection(out_path, gt_bin):
    print('Evaluating Waymo 3D detection results...')
    cmd = 'scripts/waymo_devkit/waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/' + \
        'compute_detection_metrics_main {} {}'.format(out_path, gt_bin)
    subprocess.call(cmd, shell=True)


def eval_tracking(out_path, gt_bin):
    print('Evaluating Waymo 3D tracking results...')
    cmd = 'scripts/waymo_devkit/waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/' + \
        'compute_tracking_metrics_main {} {}'.format(out_path, gt_bin)
    subprocess.call(cmd, shell=True)


def main():
    args = parse_args()

    tracking_result_path = osp.join(args.work_dir, 'output.json')
    out_path = osp.join(args.work_dir, 'result_3D.bin')

    waymo_annos = convert_track(tracking_result_path)

    create_waymo_sumbit(waymo_annos, out_path)

    if args.phase == 'val':
        eval_detection(out_path, args.gt_bin)
        eval_tracking(out_path, args.gt_bin)


if __name__ == '__main__':
    main()
