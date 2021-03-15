import argparse
from collections import defaultdict
import os
import os.path as osp
import torch

import mmcv
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel
from qd3dt.datasets import build_dataloader, build_dataset
from qd3dt.core import wrap_fp16_model
from qd3dt.models import build_detector
from qd3dt.datasets.video.bdd_eval import mmeval_by_video as tracking2d_evaluate

from tools.general_output import general_output
from scripts.plot_tracking import Visualizer
from scripts.bdd_evaluate_tracking import parse_result
from scripts.eval_argo_mot import evaluate as tracking3d_evaluate
from scripts.object_ap_eval.test_det_ap import show_result
from scripts.object_ap_eval.coco_format import read_file, load_annos
from scripts.kitti_devkit.evaluate_tracking import evaluate as kitti_evaluate

ablation_study_list = [
    'motion_comparison', 'center2d3d', 'ablation_study',
    'ablation_study_drop_feature', 'core_eval', 'match_comparison'
]

cat_mapping = {
    'kitti': ['Car', 'Pedestrian', 'Cyclist'],
    'gta': ['Car'],
    'nuscenes':
    ['Bicycle', 'Motorcycle', 'Pedestrian', 'Bus', 'Car', 'Trailer', 'Truck'],
    'waymo': ['Car', 'Pedestrian', 'Cyclist'],
}


def parse_args():
    parser = argparse.ArgumentParser(description='qd3dt test detector')
    parser.add_argument(
        'dataset_name', help='dataset name', choices=cat_mapping.keys())
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('out', help='output result file')
    parser.add_argument(
        '--data_split_prefix', help='data_split path for output log')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='track',
        choices=['track', 'bbox'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show_time', action='store_true', help='show profiling time')
    parser.add_argument(
        '--add_test_set', action='store_true', help='inference test result')
    parser.add_argument(
        '--add_ablation_exp',
        default=None,
        choices=ablation_study_list + ['all'],
        help='inference ablation study types')
    parser.add_argument(
        '--pure_det',
        action='store_true',
        help='get pure detector output for lstm training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--full_frames',
        action='store_true',
        help='using all frames imformation for nuScenes')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def run_inference_and_evaluate(args, cfg, out_path):
    print(f"Starting {out_path} ...")
    run_inference(cfg, args.checkpoint, out_path, show_time=args.show_time)
    run_evaluate(args.dataset_name, cfg, out_path)
    print()


def run_inference(cfg,
                  checkpoint_path: str,
                  out_path: str,
                  show_time: bool,
                  pure_det: bool = False):
    out_pkl_path = osp.join(out_path, 'output.pkl')
    out_json_path = osp.join(out_path, 'output.json')

    if osp.isfile(out_pkl_path):
        print(f"{out_pkl_path} already exist")
        return

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False
    dist_vid_test = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        dist_vid_test=dist_vid_test)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    model.dataset = dataset
    model.out = out_path
    model.data = cfg.data.test

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])

    coco_outputs, outputs = single_gpu_test(
        model,
        data_loader,
        out_path=out_path,
        show_time=show_time,
        modelcats=cfg.test_cfg.save_txt,
        pure_det=pure_det)

    rank, _ = get_dist_info()
    if rank == 0:
        print(f'\nwriting results to {out_path}')
        mmcv.dump(outputs, out_pkl_path)
        mmcv.dump(coco_outputs, out_json_path)


def run_evaluate(dataset_name: str, cfg, out_path: str):
    out_pkl_path = osp.join(out_path, 'output.pkl')
    out_json_path = osp.join(out_path, 'output.json')
    res_folder = osp.join(out_path, 'txts')

    category = cat_mapping[dataset_name]
    ann_file = cfg.data.test.ann_file

    if dataset_name in ['kitti']:
        print('Detection 2D results:')
        if 'DET' in ann_file:
            kitti_evaluate(
                '{}.seqmap'.format(ann_file['DET']),
                cfg.data.test.img_prefix['DET'].replace('image', 'label'),
                res_folder, cfg.test_cfg.analyze)
        else:
            kitti_evaluate(
                '{}.seqmap'.format(ann_file['VID']),
                cfg.data.test.img_prefix['VID'].replace('image', 'label'),
                res_folder, cfg.test_cfg.analyze)

    if dataset_name in ['gta', 'kitti']:
        run_3d_detection_evaluate(ann_file['VID'], out_json_path, res_folder,
                                  category)
        run_2d_tracking_evaluate(ann_file['VID'], out_pkl_path, res_folder)

    run_3d_tracking_evaluate(ann_file['VID'], out_json_path, res_folder,
                             category)


def run_3d_detection_evaluate(gt_path: str, out_json_path: str,
                              res_folder: str, category: list):
    print('Detection 3D results:')
    out_filename = osp.join(res_folder, "detection_3d_summary_41pts.txt")
    if osp.isfile(out_filename):
        print(f"{out_filename} already exist")
    else:
        print(f"output file name = {out_filename}")
        gt_annos = load_annos(gt_path)
        pd_annos = load_annos(out_json_path)
        with open(out_filename, "w") as out_file:
            show_result(
                gt_annos, pd_annos, obj_type=category, out_file=out_file)


def run_2d_tracking_evaluate(gt_path: str, out_pkl_path: str, res_folder: str):
    print('Tracking 2D results:')
    out_filename = osp.join(res_folder, "tracking_2d_summary.txt")
    if osp.isfile(out_filename):
        print(f"{out_filename} already exist")
    else:
        print(f"output file name = {out_filename}")
        track_eval = tracking2d_evaluate(
            mmcv.load(gt_path),
            mmcv.load(out_pkl_path)['track_results'],
            class_average=False)
        with open(out_filename, 'w') as f:
            f.write(parse_result(track_eval))


def run_3d_tracking_evaluate(gt_path: str, out_json_path: str, res_folder: str,
                             category: list):
    print('Tracking 3D results:')
    flag = 'tracking_3d'
    d_min_list = [0, 0, 0, 30, 50]
    d_max_list = [100, 50, 30, 50, 100]
    gt_annos = read_file(gt_path, category)
    pd_annos = read_file(out_json_path, category)
    for d_min, d_max in zip(d_min_list, d_max_list):
        out_filename = osp.join(res_folder,
                                f"{flag}_{d_min:04d}_{d_max:04d}.txt")
        if osp.isfile(out_filename):
            print(f"{out_filename} already exist")
        else:
            print(f"output file name = {out_filename}")
            with open(out_filename, "w") as out_file:
                tracking3d_evaluate(gt_annos, pd_annos, d_min, d_max, out_file)


def run_visualize(dataset_name: str,
                  cfg,
                  out_path: str,
                  category: list,
                  draw_3d: bool = True,
                  draw_2d: bool = False):
    print('Ploting 3D boxes and BEV:')
    out_json_path = osp.join(out_path, 'output.json')

    gt_annos = read_file(cfg.data.test.ann_file['VID'], category)
    pd_annos = read_file(out_json_path, category)

    fps = 7.0
    draw_bev = True
    is_save = is_merge = is_remove = True
    visualizer = Visualizer(dataset_name, out_json_path, fps, draw_bev, draw_2d, draw_3d,
                            is_save, is_merge, is_remove)
    visualizer.save_vid(pd_annos, gt_annos)


def single_gpu_test(model,
                    data_loader,
                    show_time: bool = False,
                    modelcats: dict = None,
                    out_path: str = None,
                    pure_det: bool = False):
    model.eval()
    outputs = defaultdict(list)
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    coco_outputs = defaultdict(list)
    pred_id = 0

    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result, use_3d_center = model(
                return_loss=False, rescale=True, pure_det=pure_det, **data)

        img_info = data['img_meta'][0].data[0][0]['img_info']

        if img_info.get(
                'is_key_frame') is not None and not img_info['is_key_frame']:
            prog_bar.update()
            continue

        coco_outputs, pred_id = general_output(coco_outputs, result, img_info,
                                               use_3d_center, pred_id,
                                               modelcats, out_path)
        outputs['bbox_results'].append(result['bbox_results'])
        outputs['track_results'].append(result['track_results'])
        if 'depth_results' in result:
            outputs['depth_results'].append(result['depth_results'])

        prog_bar.update()

    if show_time:
        mmcv.check_accum_time(timer_id=None, output=True)

    return coco_outputs, outputs


def best_model(args, out_path: str):
    if args.dataset_name == 'nuscenes':
        best_model_Nusc(args, out_path)
    elif args.dataset_name == 'kitti':
        best_model_KITTI(args, out_path)
    elif args.dataset_name == 'gta':
        best_model_GTA(args, out_path)
    elif args.dataset_name == 'waymo':
        best_model_Waymo(args, out_path)
    else:
        raise NotImplementedError


def best_model_KITTI(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.init_score_thr = 0.5
    cfg.test_cfg.track.nms_class_iou_thr = 0.8
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'LSTM3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_lstm_3dcen')

    if args.add_test_set:
        run_inference(cfg, args.checkpoint, out_path_exp, args.show_time)
    else:
        cfg.data.test.ann_file = cfg.data.val.ann_file
        cfg.data.test.img_prefix = cfg.data.val.img_prefix
        run_inference_and_evaluate(args, cfg, out_path_exp)

    if args.show:
        run_visualize(
            args.dataset_name,
            cfg,
            out_path_exp,
            cat_mapping[args.dataset_name],
            draw_3d=True,
            draw_2d=False)
        if args.add_test_set:
            run_visualize(
                args.dataset_name,
                cfg,
                out_path_exp,
                cat_mapping[args.dataset_name],
                draw_3d=False,
                draw_2d=True)


def best_model_Nusc(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'LSTM3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_lstm_3dcen')

    if args.add_test_set:
        if args.full_frames:
            cfg.data.test.ann_file['VID'] = ".".join([
                cfg.data.test.ann_file['VID'].split('.')[0] + '_full_frames',
                cfg.data.test.ann_file['VID'].split('.')[1]
            ])
        run_inference(cfg, args.checkpoint, out_path_exp, args.show_time)
    else:
        cfg.data.test.ann_file = cfg.data.val.ann_file
        if args.full_frames:
            cfg.data.test.ann_file['VID'] = ".".join([
                cfg.data.test.ann_file['VID'].split('.')[0] + '_full_frames',
                cfg.data.test.ann_file['VID'].split('.')[1]
            ])
        cfg.data.test.img_prefix = cfg.data.val.img_prefix
        run_inference_and_evaluate(args, cfg, out_path_exp)

    if args.show:
        run_visualize(args.dataset_name, cfg, out_path_exp, cat_mapping[args.dataset_name])


def best_model_GTA(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'LSTM3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_lstm_3dcen')

    if args.add_test_set:
        run_inference(cfg, args.checkpoint, out_path_exp, args.show_time)
    else:
        cfg.data.test.ann_file = cfg.data.val.ann_file
        cfg.data.test.img_prefix = cfg.data.val.img_prefix
        run_inference_and_evaluate(args, cfg, out_path_exp)

    if args.show:
        run_visualize(args.dataset_name, cfg, out_path_exp, cat_mapping[args.dataset_name])


def best_model_Waymo(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'LSTM3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_lstm_3dcen')

    if args.add_test_set:
        run_inference(cfg, args.checkpoint, out_path_exp, args.show_time)
    else:
        cfg.data.test.ann_file = cfg.data.val.ann_file
        cfg.data.test.img_prefix = cfg.data.val.img_prefix
        run_inference_and_evaluate(args, cfg, out_path_exp)

    if args.show:
        run_visualize(args.dataset_name, cfg, out_path_exp, cat_mapping[args.dataset_name])


def core_eval(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.test_cfg.use_3d_center = False
    cfg.test_cfg.track.with_bbox_iou = False
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = False
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'DummyTracker'
    out_path_exp = out_path.replace('output',
                                    f'output_{data_split}_deep_disable_3d')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.test_cfg.use_3d_center = False
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = False
    cfg.test_cfg.track.with_depth_ordering = False
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'DummyTracker'
    out_path_exp = out_path.replace('output', f'output_{data_split}_box3d')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.test_cfg.use_3d_center = False
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = False
    cfg.test_cfg.track.with_depth_ordering = False
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box2d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'DummyTracker'
    out_path_exp = out_path.replace('output', f'output_{data_split}_box2d')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = False
    cfg.test_cfg.track.with_deep_feat = False
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'DummyTracker'
    out_path_exp = out_path.replace('output', f'output_{data_split}_depth')
    run_inference_and_evaluate(args, cfg, out_path_exp)


def center2d3d(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 1.0
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = False
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 1.0
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_kf3d_2dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)


def ablation_study(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box2d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box2d_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box2d_depth_aware'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box2d_depth_aware_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'bev'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_bev_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'cosine'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_cosine_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'centroid'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_centroid_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'pure_motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box3d_deep_depth_pure_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)


def ablation_study_drop_feature(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = False
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_no_box3d_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = False
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_no_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = False
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_no_depth_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box3d_deep_depth_motion_no_uncertainty_kf3d_3dcen'
    )
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 1.0
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'DummyTracker'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box3d_deep_depth_motion_dummy_pure_det_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)


def motion_comparison(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'LSTM3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_lstm_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'LSTM3DTracker'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box3d_deep_depth_motion_no_uncertainty_lstm_3dcen'
    )
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box3d_deep_depth_motion_no_uncertainty_kf3d_3dcen'
    )
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'DummyTracker'
    out_path_exp = out_path.replace(
        'output', f'output_{data_split}_box3d_deep_depth_motion_dummy_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = False
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 1.0
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'DummyTracker'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box3d_deep_depth_motion_dummy_pure_det_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)


def pure_detection(args, out_path: str):
    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    if data_split == 'train':
        cfg.data.test.ann_file = cfg.data.train.ann_file
        cfg.data.test.img_prefix = cfg.data.train.img_prefix
    elif data_split == 'val':
        cfg.data.test.ann_file = cfg.data.val.ann_file
        cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.model.bbox_3d_head.use_uncertainty = True
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = False
    cfg.test_cfg.track.motion_momentum = 1.0
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'DummyTracker'
    out_path_exp = out_path.replace('output',
                                    f'output_{data_split}_pure_det_3dcen')
    run_inference(cfg, args.checkpoint, out_path_exp, args.show_time,
                  args.pure_det)


def match_comparison(args, out_path: str):
    args.show_time = True

    data_split = args.data_split_prefix

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    cfg.test_cfg.track.match_algo = 'greedy'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box3d_deep_depth_motion_greedy_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = cfg.data.val.ann_file
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.test_cfg.use_3d_center = True
    cfg.test_cfg.track.with_bbox_iou = True
    cfg.test_cfg.track.with_deep_feat = True
    cfg.test_cfg.track.with_depth_ordering = True
    cfg.test_cfg.track.with_depth_uncertainty = True
    cfg.test_cfg.track.motion_momentum = 0.9
    cfg.test_cfg.track.track_bbox_iou = 'box3d'
    cfg.test_cfg.track.depth_match_metric = 'motion'
    cfg.test_cfg.track.tracker_model_name = 'KalmanBox3DTracker'
    cfg.test_cfg.track.match_algo = 'hungarian'
    out_path_exp = out_path.replace(
        'output',
        f'output_{data_split}_box3d_deep_depth_motion_hungarian_kf3d_3dcen')
    run_inference_and_evaluate(args, cfg, out_path_exp)


def main():
    args = parse_args()

    assert args.out or args.show, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    out_path = osp.dirname(args.out)

    if args.pure_det:
        pure_detection(args, out_path)
        args.add_ablation_exp = []
    elif args.add_ablation_exp is None or args.add_test_set:
        best_model(args, out_path)
        args.add_ablation_exp = []
    elif 'all' in args.add_ablation_exp:
        args.add_ablation_exp = ablation_study_list
    else:
        args.add_ablation_exp = [args.add_ablation_exp]

    if 'motion_comparison' in args.add_ablation_exp:
        motion_comparison(args, out_path)

    if 'center2d3d' in args.add_ablation_exp:
        center2d3d(args, out_path)

    if 'ablation_study' in args.add_ablation_exp:
        ablation_study(args, out_path)

    if 'ablation_study_drop_feature' in args.add_ablation_exp:
        ablation_study_drop_feature(args, out_path)

    if 'core_eval' in args.add_ablation_exp:
        core_eval(args, out_path)

    if 'match_comparison' in args.add_ablation_exp:
        match_comparison(args, out_path)


if __name__ == '__main__':
    main()
