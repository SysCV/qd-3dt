import cv2
import mmcv
import torch
import os
import os.path as osp
import numpy as np


def camera_motion_compensation(data,
                               img_meta,
                               bboxes,
                               criteria,
                               warp_mode=cv2.MOTION_EUCLIDEAN,
                               filter_size=1):
    file_names = img_meta[0]['img_info']['filename'].split('-')
    frame_ind = int(file_names[-1].split('.')[0])
    fill = len(str(frame_ind))
    pre_file_name = file_names[-1].replace(str(frame_ind), str(frame_ind - 1).zfill(fill))
    pre_file_name = '-'.join(file_names[:-1] + [pre_file_name])
    pre_img = mmcv.imread(osp.join(data['img_prefix'],
                          pre_file_name),
                          flag='grayscale')
    cur_img = mmcv.imread(osp.join(data['img_prefix'],
                          img_meta[0]['img_info']['filename']),
                          flag='grayscale')
    pre_img = mmcv.impad(pre_img.T, data['img_scale']).T
    cur_img = mmcv.impad(cur_img.T, data['img_scale']).T

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp_matrix = cv2.findTransformECC(
            pre_img, cur_img, warp_matrix, warp_mode,
            criteria, None, filter_size)
        warp_matrix = torch.from_numpy(warp_matrix).to(bboxes.device)
        dummy = bboxes.new_ones(bboxes.size(0), 1)
        pt1s = torch.cat((bboxes[:, 0:2], dummy), dim=1)
        pt2s = torch.cat((bboxes[:, 2:4], dummy), dim=1)
        new_pt1s = torch.mm(warp_matrix, pt1s.t()).t()
        new_pt2s = torch.mm(warp_matrix, pt2s.t()).t()
        bboxes = torch.cat((new_pt1s, new_pt2s, bboxes[:, -1].view(-1, 1)), dim=1)
    except cv2.error as e:
        print(img_meta[0]['img_info'], e)
    return bboxes
