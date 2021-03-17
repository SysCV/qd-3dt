import os
import os.path as osp
from collections import defaultdict
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

import scripts.bdd_utils as bu
import scripts.tracking_utils as tu

cats_mapping = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    'truck': 4,
    'tram': 5,
    'misc': 6,
    'dontcare': 7
}
gta_merge_maps = {
    'Compacts': 'car',
    'Sedans': 'car',
    'SUVs': 'car',
    'Coupes': 'car',
    'Muscle': 'car',
    'Sports Classics': 'car',
    'Sports': 'car',
    'Super': 'car',
    'Vans': 'car',
    'Off-road': 'car',
    'Service': 'car',
    'Emergency': 'car',
    'Military': 'car',
    'Car': 'car',
    'Van': 'car',
    'Industrial': 'truck',
    'Utility': 'truck',
    'Commercial': 'truck',
    'Truck': 'truck',
    'Tram': 'tram',
    'Pedestrian': 'pedestrian',
    'Cyclist': 'cyclist',
    'Person': 'pedestrian',
    'Person_sitting': 'pedestrian',
    'Misc': 'misc',
    'DontCare': 'dontcare'
}

# 1080, 1920
data_amount_dict = {'train': 1000, 'val': 100, 'test': 400}
data_set_dict = {
    'val_mini': {
        'folder': 'val',
        'amount': data_amount_dict['val'] // 2
    },
    'val': {
        'folder': 'val',
        'amount': data_amount_dict['val']
    },
    'train': {
        'folder': 'train',
        'amount': data_amount_dict['train']
    },
    'test': {
        'folder': 'test',
        'amount': data_amount_dict['test']
    },
    'train_1percent': {
        'folder': 'train',
        'amount': 100
    },
    'train_10percent': {
        'folder': 'train',
        'amount': 10
    },
    'train_20percent': {
        'folder': 'train',
        'amount': 5
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='GTA Tracking to COCO format')
    parser.add_argument(
        "set",
        choices=list(data_set_dict.keys()) + ['all'],
        help="root directory of BDD label Json files",
    )

    return parser.parse_args()


def convert_track(data_dir, subset: str):
    gta_anno = defaultdict(list)

    set_size = data_set_dict[subset]['amount']
    subset_folder = data_set_dict[subset]['folder']
    img_dir = os.path.join(data_dir, subset_folder, 'image')
    label_dir = os.path.join(data_dir, subset_folder, 'label')

    if not osp.exists(img_dir):
        print(f"Folder {img_dir} is not found")
        return None

    if not os.path.exists(label_dir):
        label_dir = None

    vid_names = sorted([f.path for f in os.scandir(label_dir) if f.is_dir()])

    # Uniformly sample videos
    if set_size < len(vid_names):
        vid_names = vid_names[::set_size]

    # get information at boxes level. Collect dict. per box, not image.
    print(f"{subset} with {len(vid_names)} sequences")

    for k, v in cats_mapping.items():
        gta_anno['categories'].append(dict(id=v, name=k))

    img_id = 0
    global_track_id = 0
    ann_id = 0

    for vid_id, vid_name in enumerate(vid_names):
        print(f"VID {vid_id} ID: {vid_name}")

        ind2id = dict()
        trackid_maps = dict()

        fr_names = sorted([
            f.path for f in os.scandir(vid_name)
            if f.is_file() and f.name.endswith('final.json')
        ])

        if vid_name == osp.join(
                data_dir,
                'train/label/rec_10090618_snow_10h14m_x-493y-1796tox-1884y1790'
        ):
            print('Bump!')
            continue

        vid_info = dict(id=vid_id, name=vid_name, n_frames=len(fr_names))
        gta_anno['videos'].append(vid_info)

        init_position = bu.load_json(fr_names[0])['extrinsics']['location']

        for fr_idx, fr_name in enumerate(fr_names):
            frame = bu.load_json(fr_name)

            img_name = fr_name.replace('label', 'image').replace('json', 'jpg')
            height = frame['resolution']['height']
            width = frame['resolution']['width']
            rot_angle = np.array(frame['extrinsics']['rotation'])
            rot_matrix = tu.angle2rot(rot_angle)
            gps_to_camera = tu.angle2rot(np.array([np.pi / 2, 0, 0]),
                                         inverse=True)
            rot_matrix = rot_matrix.dot(gps_to_camera)
            rotation = R.from_matrix(rot_matrix).as_euler('xyz')
            position = [
                float(p_t) - float(p_0) for (
                    p_t,
                    p_0) in zip(frame['extrinsics']['location'], init_position)
            ]
            pose_dict = dict(rotation=rotation.tolist(), position=position)

            projection = np.array(frame['intrinsics']['cali'])

            index = fr_idx
            img_info = dict(file_name=img_name,
                            cali=projection.tolist(),
                            pose=pose_dict,
                            height=height,
                            width=width,
                            fov=60,
                            near_clip=0.15,
                            timestamp=frame['timestamp'],
                            id=img_id,
                            video_id=vid_id,
                            index=index)

            gta_anno['images'].append(img_info)

            ind2id[index] = img_id
            img_id += 1

            for label in frame['labels']:
                cat = label['category']
                if cat in ['DontCare']:
                    continue
                image_id = ind2id[index]
                if label['id'] in trackid_maps.keys():
                    track_id = trackid_maps[label['id']]
                else:
                    track_id = global_track_id
                    trackid_maps[label['id']] = track_id
                    global_track_id += 1
                x1, y1, x2, y2 = float(label['box2d']['x1']), float(
                    label['box2d']['y1']), float(label['box2d']['x2']), float(
                        label['box2d']['y2'])
                location = bu.get_label_array([label], ['box3d', 'location'],
                                              (0, 3)).astype(float)
                center_2d = tu.cameratoimage(location,
                                             projection).flatten().tolist()
                ann = dict(id=ann_id,
                           image_id=image_id,
                           category_id=cats_mapping[gta_merge_maps[cat]],
                           instance_id=track_id,
                           alpha=float(label['box3d']['alpha']),
                           roty=float(label['box3d']['orientation']),
                           dimension=[
                               float(dim)
                               for dim in label['box3d']['dimension']
                           ],
                           translation=[
                               float(loc) for loc in label['box3d']['location']
                           ],
                           is_occluded=int(label['attributes']['occluded']),
                           is_truncated=int(label['attributes']['truncated']),
                           center_2d=center_2d,
                           delta_2d=[
                               center_2d[0] - (x1 + x2) / 2.0,
                               center_2d[1] - (y1 + y2) / 2.0
                           ],
                           bbox=[x1, y1, x2 - x1, y2 - y1],
                           area=(x2 - x1) * (y2 - y1),
                           iscrowd=False,
                           ignore=label['attributes']['ignore'],
                           segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]])
                gta_anno['annotations'].append(ann)
                ann_id += 1
    return gta_anno


def main():

    args = parse_args()

    data_dir = 'data/GTA/'
    out_dir = 'data/GTA/anns/'

    print('Convert GTA Tracking dataset to COCO style.')
    if not osp.isfile(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if args.set == 'all':
        for set_name in data_set_dict:
            print(osp.join(data_dir, set_name))
            ann = convert_track(data_dir, set_name)
            bu.dump_json(osp.join(out_dir, f'tracking_{set_name}.json'), ann)
    else:
        print(osp.join(data_dir, args.set))
        ann = convert_track(data_dir, args.set)
        bu.dump_json(osp.join(out_dir, f'tracking_{args.set}.json'), ann)


if __name__ == "__main__":
    main()
