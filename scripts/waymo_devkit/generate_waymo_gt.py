import os
import os.path as osp
import math
import json
from tqdm import tqdm
from collections import defaultdict
import tensorflow.compat.v1 as tf

from PIL import Image

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

lidar_list = [
    '_FRONT', '_FRONT_LEFT', '_FRONT_RIGHT', '_SIDE_LEFT', '_SIDE_RIGHT'
]


def create_gt(data_dir, out_dir, phase: str, mini=False):

    raw_dir = osp.join(data_dir, 'raw')

    dataset = tf.data.TFRecordDataset([
        osp.join(raw_dir, phase, p)
        for p in os.listdir(osp.join(raw_dir, phase))
        if p.endswith('.tfrecord')
    ],
                                      compression_type='')

    objects = metrics_pb2.Objects()
    projected_objects = metrics_pb2.Objects()

    mini_seq_id = 0
    cur_video = ''
    fr_id = 0

    for data in tqdm(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        id_to_bbox = dict()
        id_list = []

        if not cur_video == frame.context.name:
            if mini:
                if mini_seq_id < 1:
                    mini_seq_id += 1
                else:
                    break
            cur_video = frame.context.name
            fr_id = 0

        for labels in frame.projected_lidar_labels:
            for label in labels.labels:
                id_to_bbox[label.id] = 1

        for obj in frame.laser_labels:
            # type = 0 UNKNOWN, type = 3 SIGN
            if obj.type == 0 or obj.type == 3:
                continue

            if obj.num_lidar_points_in_box < 1:
                continue

            o = metrics_pb2.Object()
            o.context_name = cur_video
            o.frame_timestamp_micros = frame.timestamp_micros

            box = label_pb2.Label.Box()
            box.center_x = obj.box.center_x
            box.center_y = obj.box.center_y
            box.center_z = obj.box.center_z
            box.length = obj.box.length
            box.width = obj.box.width
            box.height = obj.box.height
            box.heading = obj.box.heading
            o.object.box.CopyFrom(box)

            o.score = 1.0

            o.object.type = obj.type
            o.object.id = obj.id
            o.object.num_lidar_points_in_box = obj.num_lidar_points_in_box

            for lidar in lidar_list:
                if obj.id + lidar in list(id_to_bbox.keys()) and not(obj.id in id_list):
                    id_list.append(obj.id)
                    projected_objects.objects.append(o)

            objects.objects.append(o)

        fr_id += 1

    if mini:
        projected_out_path = osp.join(out_dir, 'gt_mini_projected.bin')
        out_path = osp.join(out_dir, 'gt_mini.bin')
    else:
        projected_out_path = osp.join(out_dir, f'gt_{phase}_projected.bin')
        out_path = osp.join(out_dir, f'gt_{phase}.bin')

    # Write objects to a file.
    with open(projected_out_path, 'wb') as f:
        f.write(projected_objects.SerializeToString())

    with open(out_path, 'wb') as f:
        f.write(objects.SerializeToString())


def main():

    data_dir = 'data/Waymo'
    out_dir = 'data/Waymo'

    print('Convert Waymo Tracking dataset to gt.bin.')

    print("converting mini")
    create_gt(data_dir, out_dir, "validation", mini=True)

    print("converting validation")
    create_gt(data_dir, out_dir, "validation")


if __name__ == "__main__":
    main()