import os
import os.path as osp
import math
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from scipy.spatial.transform import Rotation as R
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset
from nuscenes.utils.geometry_utils import view_points

import scripts.bdd_utils as bu

id_camera_dict = {
    'front': 1,
    'front_left': 2,
    'front_right': 3,
    'side_left': 4,
    'side_right': 5
}

lidar_list = [
    '_FRONT', '_FRONT_LEFT', '_FRONT_RIGHT', '_SIDE_LEFT', '_SIDE_RIGHT'
]

cats_mapping = {'pedestrian': 1, 'cyclist': 2, 'car': 3, 'dontcare': 4}

waymo_cats = {'2': 'pedestrian', '1': 'car', '4': 'cyclist'}

instance_id_dict = dict()


def get_box_transformation_matrix(obj_loc, obj_size, ry):
    """Create a transformation matrix for a given label box pose."""

    tx, ty, tz = obj_loc
    c = math.cos(ry)
    s = math.sin(ry)

    sl, sh, sw = obj_size

    return np.array([[sl * c, -sw * s, 0, tx], [sl * s, sw * c, 0, ty],
                     [0, 0, sh, tz], [0, 0, 0, 1]])


def rot_y2alpha(rot_y, x, FOCAL_LENGTH):
    """
    Get alpha by rotation_y - theta
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x, FOCAL_LENGTH)
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    return alpha


def _bbox_inside(box1, box2):
    return box1[0] > box2[0] and box1[0] + box1[2] < box2[0] + box2[2] and \
            box1[1] > box2[1] and box1[1] + box1[3] < box2[1] + box2[3]


def convert_track(data_dir, phase: str, mini=False):

    raw_dir = osp.join(data_dir, 'raw')

    if not osp.exists(osp.join(raw_dir, phase)):
        print(f"Folder {osp.join(raw_dir, phase)} is not found")
        return None

    coco_json = defaultdict(list)

    for k, v in cats_mapping.items():
        coco_json['categories'].append(dict(id=v, name=k))

    dataset = tf.data.TFRecordDataset([
        osp.join(raw_dir, phase, p)
        for p in os.listdir(osp.join(raw_dir, phase))
        if p.endswith('.tfrecord')
    ],
                                      compression_type='')

    for cam in id_camera_dict.keys():
        mini_seq_id = 0
        fr_id = 0
        cur_video = ''
        for data in tqdm(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            id_to_bbox = dict()

            # save video info
            if not cur_video == frame.context.name:
                if mini:
                    if mini_seq_id < 1:
                        mini_seq_id += 1
                    else:
                        break
                cur_video = frame.context.name
                coco_json['videos'].append(
                    dict(id=len(coco_json['videos']),
                         name=f'{cur_video}_{id_camera_dict[cam]}'))
                fr_id = 0

            i_dict = dict()

            # save images & labels
            for i in frame.images:
                i_dict[i.name] = i

            for camera, labels in zip(
                    frame.context.camera_calibrations,
                    frame.projected_lidar_labels
                    if phase != 'testing' else [None for _ in frame.images]):

                if camera.name != id_camera_dict[cam]:
                    continue

                video_base_dir = osp.join(data_dir, 'images_png', cam, phase,
                                          cur_video)

                img_name = osp.join(video_base_dir,
                                    '{}_{:07d}.png'.format(cur_video, fr_id))
                img_array = tf.image.decode_jpeg(
                    i_dict[camera.name].image).numpy()
                img = Image.fromarray(img_array)

                del i_dict

                if not osp.isfile(img_name):
                    os.makedirs(video_base_dir, exist_ok=True)
                    # save image: $data_dir/images_png/$camera/$phase/$video_name
                    img.save(img_name)

                # pose
                global_from_car = np.array(frame.pose.transform).reshape(4, 4)

                # camera extrinsic
                car_from_cam = np.array(camera.extrinsic.transform).reshape(
                    4, 4)
                cam_from_car = np.linalg.inv(car_from_cam)
                waymo2kitti_RT = np.array([[0, -1, 0, 0], [0, 0, -1, 0],
                                           [1, 0, 0, 0], [0, 0, 0, 1]])

                cam_from_car = np.dot(waymo2kitti_RT, cam_from_car)

                car_from_cam = np.linalg.inv(cam_from_car)

                tm = np.dot(global_from_car, car_from_cam)

                rotation_matrix = tm[:3, :3]
                position = tm[:3, 3].tolist()

                rotation = R.from_matrix(rotation_matrix).as_euler(
                    'xyz').tolist()
                pose_dict = dict(rotation=rotation, position=position)

                # camera intrinsic
                camera_intrinsic = np.zeros((3, 4))
                camera_intrinsic[0, 0] = camera.intrinsic[0]
                camera_intrinsic[1, 1] = camera.intrinsic[1]
                camera_intrinsic[0, 2] = camera.intrinsic[2]
                camera_intrinsic[1, 2] = camera.intrinsic[3]
                camera_intrinsic[2, 2] = 1

                camera_intrinsic = camera_intrinsic.tolist()

                coco_json['images'].append(
                    dict(
                        file_name=img_name,
                        cali=camera_intrinsic,
                        pose=pose_dict,
                        car_from_cam=car_from_cam.tolist(),
                        height=img.size[1],
                        width=img.size[0],
                        fov=60,
                        near_clip=0.15,
                        id=len(coco_json['images']),
                        video_id=coco_json['videos'][-1]['id'],
                        timestamp=frame.timestamp_micros,
                        index=fr_id,
                    ))

                # save label
                if phase == 'testing':
                    continue

                for label in labels.labels:
                    bbox = [
                        label.box.center_x - label.box.length / 2,
                        label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2,
                        label.box.center_y + label.box.width / 2
                    ]
                    id_to_bbox[label.id] = bbox

                anns = []

                for obj in frame.laser_labels:
                    # type = 0 UNKNOWN, type = 3 SIGN
                    if obj.type == 0 or obj.type == 3:
                        continue

                    # caculate bounding box
                    bounding_box = None
                    name = None
                    for lidar in lidar_list:
                        if obj.id + lidar in id_to_bbox:
                            if obj.id + lidar not in instance_id_dict:
                                instance_id_dict[obj.id +
                                                 lidar] = len(instance_id_dict)
                            x1, y1, x2, y2 = id_to_bbox.get(obj.id + lidar)
                            x = obj.box.center_x
                            y = obj.box.center_y
                            z = obj.box.center_z
                            h = obj.box.height
                            w = obj.box.width
                            l = obj.box.length
                            rot_y = obj.box.heading

                            transform_box_to_cam = cam_from_car @ get_box_transformation_matrix(
                                (x, y, z), (l, h, w), rot_y)
                            pt1 = np.array([-0.5, 0.5, 0, 1.])
                            pt2 = np.array([0.5, 0.5, 0, 1.])
                            pt1 = np.matmul(transform_box_to_cam, pt1).tolist()
                            pt2 = np.matmul(transform_box_to_cam, pt2).tolist()

                            new_loc = np.matmul(cam_from_car,
                                                np.array([x, y, z,
                                                          1]).T).tolist()
                            x, y, z = new_loc[:3]
                            rot_y = -math.atan2(pt2[2] - pt1[2],
                                                pt2[0] - pt1[0])
                            alpha = rot_y2alpha(rot_y, x, z).item()

                            # project 3d center to 2d.
                            center_2d = view_points(
                                np.array([x, y, z]).reshape(3, 1),
                                np.array(camera_intrinsic),
                                True).T[:, :2].tolist()[0]

                            ann = dict(
                                image_id=coco_json['images'][-1]['id'],
                                category_id=cats_mapping[waymo_cats[str(
                                    obj.type)]],
                                instance_id=instance_id_dict[obj.id + lidar],
                                alpha=alpha,
                                roty=rot_y,
                                dimension=[float(dim) for dim in [h, w, l]],
                                translation=[float(loc) for loc in [x, y, z]],
                                is_occluded=False,
                                is_truncated=False,
                                bbox=[x1, y1, x2 - x1, y2 - y1],
                                area=(x2 - x1) * (y2 - y1),
                                center_2d=center_2d,
                                delta_2d=[
                                    center_2d[0] - (x1 + x2) / 2.0,
                                    center_2d[1] - (y1 + y2) / 2.0
                                ],
                                iscrowd=False,
                                ignore=False,
                                segmentation=[[x1, y1, x1, y2, x2, y2, x2,
                                               y1]])
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
                    ann['id'] = len(coco_json['annotations'])
                    coco_json['annotations'].append(ann)

            fr_id += 1
    return coco_json


def main():

    data_dir = 'data/Waymo'
    out_dir = 'data/Waymo/anns'

    print('Convert Waymo Tracking dataset to COCO style.')
    if not osp.isfile(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("tracking mini")
    coco_json = convert_track(data_dir, "validation", mini=True)
    bu.dump_json(osp.join(out_dir, 'tracking_val_mini.json'), coco_json)

    print("tracking train")
    coco_json = convert_track(data_dir, "training")
    bu.dump_json(osp.join(out_dir, 'tracking_train.json'), coco_json)

    print("tracking validation")
    coco_json = convert_track(data_dir, "validation")
    bu.dump_json(osp.join(out_dir, 'tracking_val.json'), coco_json)

    print("tracking testing")
    coco_json = convert_track(data_dir, "testing")
    bu.dump_json(osp.join(out_dir, 'tracking_test.json'), coco_json)


if __name__ == "__main__":
    main()
