import os.path as osp
import sys
import argparse
import pickle as pkl

import mmcv

from qd3dt.datasets.video.bdd_eval import mmeval_by_video as bdd_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description='Monocular 3D Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gt_map', help='groundtruth seqmap')
    parser.add_argument('gt_folder', help='groundtruth label folder')
    parser.add_argument('res_folder', help='results folder')
    args = parser.parse_args()

    return args


def parse_result(track_result: dict):
    """
    Parse track result to string
    """
    lines = ""
    for metric_key in track_result:
        lines += f"{metric_key},"
        for class_key in track_result[metric_key]:
            lines += f"{track_result[metric_key][class_key]},"
        lines += "\n"
    return lines


def main():
    args = parse_args()

    gt_annos = mmcv.load(args.gt_folder)
    with open(osp.join(args.res_folder, '..', 'output.pkl'), 'rb') as f:
        pd_annos = pkl.load(f)['track_results']

    track_eval = bdd_eval(gt_annos, pd_annos, class_average=False)

    with open(osp.join(args.res_folder, 'tracking_2d_summary.txt'), 'w') as f:
        f.write(parse_result(track_eval))


if __name__ == '__main__':
    main()
