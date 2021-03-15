import os.path as osp
import argparse

from scripts.object_ap_eval import kitti_common as kitti
from scripts.object_ap_eval import coco_format as cf
from scripts.object_ap_eval.eval import get_official_eval_result, get_coco_eval_result
"""
python scripts/object_ap_eval/test_det_ap.py data/GTA/anns/tracking_mini.json.seqmap data/GTA/anns/tracking_mini.json work_dirs/GTA/quasi_r50_dcn_3dmatch_multibranch_conv_clsreg/output/txts
python scripts/object_ap_eval/test_det_ap.py data/KITTI/anns/tracking_all-C_subval.json.seqmap data/KITTI/anns/tracking_all-C_subval.json work_dirs/KITTI/quasi_r50_dcn_3dmatch_multibranch_conv_dep_dim_rot9/output/txts
python scripts/object_ap_eval/test_det_ap.py data/KITTI/anns/detection_7cls_subtrain.json.seqmap data/KITTI/detection/training/label_2 work_dirs/KITTI/quasi_r50_dcn_3dmatch_multibranch_conv_dep_dim_rot13/output_det_subtrain/txts
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Monocular 3D Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gt_map', help='groundtruth seqmap')
    parser.add_argument('gt_folder', help='groundtruth label folder')
    parser.add_argument('res_folder', help='results folder')
    parser.add_argument(
        '--conf_threshold',
        default=-1,
        type=float,
        help='Filter low confidence predictions')
    args = parser.parse_args()

    return args


def get_frm_map(path_seq_map: str):
    with open(path_seq_map, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def get_seq_map(path_seq_map: str):
    """get #sequences and #frames per sequence from test mapping
    """
    n_frames = []
    sequence_name = []
    with open(path_seq_map, "r") as fh:
        for lines in fh:
            fields = lines.split(" ")
            sequence_name.append(fields[0])
            n_frames.append(int(fields[3]) - int(fields[2]) + 1)
    return sequence_name, n_frames


def show_result(gt_annos, pd_annos, obj_type='Car', out_file=None):
    # obj_type [0] == Car

    result_kitti = get_official_eval_result(gt_annos, pd_annos, obj_type)
    print(result_kitti)

    result_coco = get_coco_eval_result(gt_annos, pd_annos, obj_type)
    print(result_coco)

    if out_file is not None:
        out_file.write(result_kitti)
        out_file.write('\n')
        out_file.write(result_coco)


def main():
    args = parse_args()

    if 'detection' in args.gt_map:
        val_image_ids = get_frm_map(args.gt_map)
        pd_annos = kitti.get_label_annos(args.res_folder, val_image_ids)
        if args.conf_threshold > 0:
            pd_annos = kitti.filter_annos_low_score(pd_annos,
                                                    args.conf_threshold)
        gt_annos = kitti.get_label_annos(args.gt_folder, val_image_ids)
    else:
        # seq_lists, n_frames = get_seq_map(args.gt_map)
        # pd_annos = kitti.get_label_annos_by_folder(
        #     args.res_folder, seq_lists, n_frames, adjust_center=False)
        gt_annos = cf.load_annos(args.gt_folder)
        pd_annos = cf.load_annos(args.res_folder)

    out_filename = osp.join(
        osp.dirname(args.res_folder), "detection_3d_summary.txt")
    print(f"output file name = {out_filename}")
    # Show KITTI, COCO evaluation results
    with open(out_filename, "w") as out_file:
        show_result(
            gt_annos,
            pd_annos,
            obj_type=['Car', 'Pedestrian', 'Cyclist'],
            out_file=out_file)


if __name__ == '__main__':
    main()
