'''
load annotation from json files in COCO-ish format
'''
import json
import numpy as np


def xywh2xyxy(bbox: list):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def xywh2center(bbox: list):
    return [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0]


def load_annos(path):

    print("Loading GT file {} ...".format(path))
    coco_annos = json.load(open(path, 'r'))

    assert len(coco_annos) > 0, "{} has no files".format(path)

    seq_anno = []
    obj_num = 0
    map_cat_dict = {
        cat_dict['id']: cat_dict['name'].capitalize()
        for cat_dict in coco_annos['categories']
    }
    n_frame = len(coco_annos['images'])
    obj_dicts = coco_annos['annotations']

    for fr_id in range(n_frame):
        # print(
        #     f"{fr_id}/{n_frame}, {obj_num}/{len(obj_dicts)}, {len(seq_anno)}")

        frm_anno = {
            'fr_id': [],
            'track_id': [],
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'box_center': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': []
        }

        while obj_num < len(obj_dicts):
            obj_dict = obj_dicts[obj_num]

            if obj_dict['image_id'] != fr_id:
                break

            frm_anno['fr_id'].append(obj_dict['image_id'])
            frm_anno['track_id'].append(obj_dict['instance_id'])
            frm_anno['name'].append(map_cat_dict[obj_dict['category_id']])
            frm_anno['truncated'].append(obj_dict['is_truncated'])
            frm_anno['occluded'].append(obj_dict['is_occluded'])
            frm_anno['bbox'].append([
                obj_dict['bbox'][0], obj_dict['bbox'][1],
                obj_dict['bbox'][0] + obj_dict['bbox'][2],
                obj_dict['bbox'][1] + obj_dict['bbox'][3]
            ])
            frm_anno['alpha'].append(obj_dict['alpha'])
            frm_anno['dimensions'].append(obj_dict['dimension'])
            frm_anno['location'].append(obj_dict['translation'])
            frm_anno['rotation_y'].append(obj_dict['roty'])
            frm_anno['box_center'].append(
                obj_dict.get('center_2d', xywh2center(obj_dict['bbox'])))
            frm_anno['score'].append(1.0 if not (
                'score' in obj_dict.keys()) else obj_dict['score'])

            obj_num += 1

        frm_anno['fr_id'] = np.array(frm_anno['fr_id'])
        frm_anno['track_id'] = np.array(frm_anno['track_id'])
        frm_anno['name'] = np.array(frm_anno['name'])
        frm_anno['truncated'] = np.array(frm_anno['truncated']).astype(float)
        frm_anno['occluded'] = np.array(frm_anno['occluded']).astype(int)
        frm_anno['alpha'] = np.array(frm_anno['alpha']).astype(float)
        frm_anno['bbox'] = np.array(frm_anno['bbox']).astype(float).reshape(
            -1, 4)
        # dimensions will convert hwl order to lhw order for evalaution
        frm_anno['dimensions'] = np.array(
            frm_anno['dimensions']).astype(float).reshape(-1, 3)[:, [2, 0, 1]]
        # location will keep xyz order for evaluation
        frm_anno['location'] = np.array(
            frm_anno['location']).astype(float).reshape(-1, 3)
        frm_anno['rotation_y'] = np.array(
            frm_anno['rotation_y']).astype(float).reshape(-1)
        frm_anno['score'] = np.array(
            frm_anno['score']).astype(float).reshape(-1)

        seq_anno.append(frm_anno.copy())

    print("GT file {} loaded".format(path))
    return seq_anno


def read_file(path: str, category: list):
    """Load dictionary from file.

    Args:
        fpath: Path to file.

    Returns:
        Deserialized Python dictionary.
    """

    print("Loading GT file {} ...".format(path))
    coco_annos = json.load(open(path, 'r'))

    assert len(coco_annos) > 0, "{} has no files".format(path)

    log_anno_dict = {}
    # obj_num = 0
    # total_frame = len(coco_annos['images'])
    # total_annotations = len(coco_annos['annotations'])

    map_cat_dict = {
        cat_dict['id']: cat_dict['name'].capitalize()
        for cat_dict in coco_annos['categories']
    }

    for seq_num, log_dict in enumerate(coco_annos['videos']):
        log_id = log_dict['id']
        # print(f"{log_dict['id']} {log_name}")

        log_anno_dict[log_id] = {
            'frames': {},
            'seq_name': log_dict['name'],
            'width': None,
            'height': None
        }

    for fr_num, img_dict in enumerate(coco_annos['images']):
        # print(f"{fr_num}/{total_frame}")

        frm_anno_dict = {
            'cam_loc': img_dict['pose']['position'],
            'cam_rot': img_dict['pose']['rotation'],
            'cam_calib': img_dict['cali'],
            'im_path': img_dict['file_name'],
            'annotations': []
        }
        log_id = img_dict['video_id']
        fr_id = img_dict['index']
        log_anno_dict[log_id]['frames'][fr_id] = frm_anno_dict.copy()
        if fr_id == 0:
            log_anno_dict[log_id]['width'] = img_dict['width']
            log_anno_dict[log_id]['height'] = img_dict['height']

    for obj_num, obj_dict in enumerate(coco_annos.get('annotations', [])):
        if map_cat_dict[obj_dict['category_id']] not in category:
            continue

        log_id = coco_annos['images'][obj_dict['image_id']]['video_id']
        fr_id = coco_annos['images'][obj_dict['image_id']]['index']
        frm_anno_dict = log_anno_dict[log_id]['frames'][fr_id]
        # print(f"{obj_num}/{total_annotations}")
        t_data = {
            'fr_id': obj_dict['image_id'],
            'track_id': obj_dict['instance_id'],
            'obj_type': map_cat_dict[obj_dict['category_id']],
            'truncated': obj_dict['is_truncated'],
            'occluded': obj_dict['is_occluded'],
            'alpha': obj_dict['alpha'],
            'box': xywh2xyxy(obj_dict['bbox']),
            'box_center': obj_dict.get('center_2d',
                                       xywh2center(obj_dict['bbox'])),
            'dimension': obj_dict['dimension'],
            'location': obj_dict['translation'],
            'yaw': obj_dict['roty'],
            'confidence': obj_dict.get('uncertainty', 0.95),
            'score': obj_dict.get('score', 1.0)
        }
        if len(frm_anno_dict['annotations']) > 0:
            frm_anno_dict['annotations'].append(t_data.copy())
        else:
            frm_anno_dict['annotations'] = [t_data.copy()]

    return log_anno_dict
