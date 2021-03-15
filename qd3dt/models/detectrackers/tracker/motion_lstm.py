import argparse
import time
import multiprocessing
import os.path as osp
import pickle
from os import mkdir
from pyquaternion import Quaternion

from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import scripts.tracking_utils as tu
import scripts.network_utils as nu
import scripts.kitti_utils as ku
from scripts.object_ap_eval.coco_format import read_file
from scripts.plot_utils import plot_3D
from motion_model import get_lstm_model, LSTM_MODEL_ZOO
from tracker_model import get_tracker, TRACKER_MODEL_ZOO
'''
CUDA_VISIBLE_DEVICES=1 python motion_lstm.py nuscenes train \
--session batch128_min10_seq10_dim7_VeloLSTM \
--min_seq_len 10 --seq_len 10 \
--lstm_model_name VeloLSTM --tracker_model_name KalmanBox3DTracker \
--input_gt_path data/nuscenes/anns/tracking_train.json \
--input_pd_path data/nuscenes/anns/tracking_output_train.json \
--cache_name work_dirs/LSTM/nuscenes_train_pure_det_min10.pkl \
--loc_dim 7 -b 128 --is_plot --show_freq 500
'''
'''
CUDA_VISIBLE_DEVICES=0 python motion_lstm.py nuscenes test \
--session batch128_min10_seq10_dim7_VeloLSTM \
--min_seq_len 10 --seq_len 10 \
--lstm_model_name VeloLSTM --tracker_model_name LSTM3DTracker \
--input_gt_path data/nuscenes/anns/tracking_val.json \
--input_pd_path data/nuscenes/anns/tracking_output_val.json \
--cache_name work_dirs/LSTM/nuscenes_val_pure_det_min10.pkl \
--num_epochs 100 --loc_dim 7 -b 1 --is_plot 
'''

np.random.seed(777)
torch.manual_seed(100)


def verbose(sentence: str, is_verbose: bool = False):
    if is_verbose:
        tqdm.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {sentence}")


def fix_alpha(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


cat_mapping = {
    'kitti': ['Car', 'Pedestrian', 'Cyclist'],
    'gta': ['Car'],
    'nuscenes': [
        'Bicycle', 'Motorcycle', 'Pedestrian', 'Bus', 'Car', 'Trailer',
        'Truck', 'Construction_vehicle', 'Traffic_cone', 'Barrier'
    ],
    'waymo': ['Car', 'Pedestrian', 'Cyclist'],
}


class SeqDataset(Dataset):

    def __init__(self,
                 dataset,
                 input_gt_path,
                 input_pd_path,
                 is_train,
                 seq_len,
                 min_seq_len,
                 max_depth,
                 depth_scale,
                 cache_name,
                 r_var=0.1):
        self.dataset = dataset
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.is_train = is_train
        self.is_pred = True
        self.r_var = r_var
        if not is_train:
            input_gt_path = [
                path.replace('train', 'val') for path in input_gt_path
            ]
            input_pd_path = [
                path.replace('_train', '_val') for path in input_pd_path
            ]
            cache_name = cache_name.replace('_train', '_val')

        if cache_name and osp.isfile(cache_name):
            verbose("Loading {} ...".format(cache_name), True)
            data = pickle.load(open(cache_name, 'rb'))
            self.tracking_seqs, self.data_key = data
        else:
            self.tracking_seqs = []
            self.data_key = []
            for pidx, (path_gt, path_pd) in enumerate(
                    zip(input_gt_path, input_pd_path)):
                assert osp.isfile(path_gt), path_gt
                assert osp.isfile(path_pd), path_pd
                verbose(f"{pidx}, {path_gt}, {path_pd}", True)
                cats = cat_mapping.get(self.dataset)
                sequences_gt = read_file(path_gt, category=cats)
                if self.is_pred:
                    sequences_pd = read_file(path_pd, category=cats)
                else:
                    sequences_pd = sequences_gt
                for iter_idx, ((seq_idx, sequence_gt),
                               (seq_idx_pd, sequence_pd)) in enumerate(
                                   zip(sequences_gt.items(),
                                       sequences_pd.items())):
                    verbose(
                        f"{iter_idx} | {seq_idx} - {sequence_gt['seq_name']} | {seq_idx_pd} - {sequence_pd['seq_name']}",
                        True)
                    self.tracking_seqs.append(
                        self.convert_tracking_pred(sequence_gt, sequence_pd))
                    self.data_key.append(
                        self.sample(self.tracking_seqs[iter_idx]))
        self.seqs_len = [len(keys) for keys in self.data_key]
        self.accum_len = np.cumsum(self.seqs_len) - 1
        self.data_len = sum(self.seqs_len)

        if cache_name and not osp.isfile(cache_name):
            if not osp.isdir(osp.dirname(cache_name)):
                mkdir(osp.dirname(cache_name))
            with open(cache_name, 'wb') as f:
                pickle.dump([self.tracking_seqs, self.data_key], f)

    def __getitem__(self, index):
        seq = np.sum(self.accum_len < index)
        fr = index - (self.accum_len[seq] if seq > 0 else 0)
        key = self.data_key[seq][fr]
        return self.get_traj_from_data_seq(seq, key)

    def __len__(self):
        return self.data_len

    def get_traj_from_data_seq(self, seq, key):
        # Dataloading
        trajectory_seq = self.tracking_seqs[seq][key]
        traj_len = max(len(trajectory_seq) - self.seq_len, 1)

        # Random sample data
        idx = 0
        if self.is_train:
            idx = np.random.randint(traj_len)
        upper_idx = self.seq_len + idx
        data_seq = trajectory_seq[idx:upper_idx]

        # Get data
        depth_gt = np.array([fr['depth_gt'] for fr in data_seq])
        alpha_gt = np.array([fr['alpha_gt'] for fr in data_seq])
        yaw_gt = np.array([fr['yaw_gt'] for fr in data_seq])
        dim_gt = np.array([fr['dim_gt'] for fr in data_seq])
        cen_gt = np.array([fr['center_gt'] for fr in data_seq])
        cam_calib = np.array(
            [np.array(fr['cam_calib']).reshape(3, 4) for fr in data_seq])
        cam_loc = np.array([fr['cam_loc'] for fr in data_seq])
        cam_rot = np.array([fr['cam_rot'] for fr in data_seq])
        pose = [
            ku.Pose(np.array(fr['cam_loc']), np.array(fr['cam_rot']))
            for fr in data_seq
        ]

        if self.is_pred:
            confidence = np.array([fr['confidence_pd'] for fr in data_seq])
            depth_random = np.array([fr['depth_pd'] for fr in data_seq])
            alpha_random = np.array([fr['alpha_pd'] for fr in data_seq])
            yaw_random = np.array([fr['yaw_pd'] for fr in data_seq])
            dim_random = np.array([fr['dim_pd'] for fr in data_seq])
            cen_random = np.array([fr['center_pd'] for fr in data_seq])

        else:
            if self.depth_scale > 0.0:
                randomness = np.random.normal(
                    0.0, self.r_var, size=depth_gt.shape)
                randomness *= (
                    np.random.rand(*depth_gt.shape) > np.exp(
                        -depth_gt / (self.depth_scale**2)))
            else:
                randomness = np.zeros(depth_gt.shape)

            confidence = np.exp(-np.abs(randomness))

            depth_random = depth_gt * (1.0 + randomness)
            yaw_random = yaw_gt * (1.0 + randomness)
            cen_random = cen_gt * (1.0 + randomness[..., None])
            dim_random = dim_gt * (1.0 + randomness[..., None]**2)

            alpha_random = alpha_gt.copy()
            rand_thrs = abs(randomness) > 2 * self.r_var
            alpha_random[rand_thrs] += np.pi
            alpha_random[np.bitwise_not(rand_thrs)] *= (
                1.0 + randomness[np.bitwise_not(rand_thrs)]**2)
            alpha_random = (alpha_random + np.pi) % (2 * np.pi) - np.pi

        # objects center in the world coordinates
        # X to the east, Y to the north, Z to the sky
        def get_box_obj(depth, alpha, dim, cen, cam_calib, pose) -> np.ndarray:
            objs_list = []
            roll_pitch_list = []
            for i in range(len(depth)):
                loc_cam = tu.imagetocamera(cen[i:i + 1], depth[i:i + 1],
                                           cam_calib[i])
                yaw = tu.alpha2rot_y(alpha[i:i + 1], loc_cam[:, 0:1],
                                     loc_cam[:, 2:3])
                quat_yaw = Quaternion(axis=[0, 1, 0], radians=yaw)
                quat_cam_rot = Quaternion(matrix=pose[i].rotation)
                quat_yaw_world = quat_cam_rot * quat_yaw
                if quat_yaw_world.z < 0:
                    quat_yaw_world *= -1
                roll_world, pitch_world, yaw_world = tu.quaternion_to_euler(
                    quat_yaw_world.w, quat_yaw_world.x, quat_yaw_world.y,
                    quat_yaw_world.z)
                loc_glb = tu.cameratoworld(loc_cam, pose[i])
                roll_pitch_list.append([roll_world, pitch_world])

                objs_list.append(
                    np.hstack([loc_glb,
                               np.array([[yaw_world]]),
                               dim[i:i + 1]]).flatten())
            return np.array(objs_list), np.array(roll_pitch_list)

        objs_gt, yaw_axis_gt = get_box_obj(depth_gt, alpha_gt, dim_gt, cen_gt,
                                           cam_calib, pose)
        objs_obs, yaw_axis_pd = get_box_obj(depth_random, alpha_random,
                                            dim_random, cen_gt, cam_calib,
                                            pose)

        # Padding
        valid_mask = np.hstack(
            [np.ones(len(objs_gt)),
             np.zeros([self.seq_len])])[:self.seq_len]
        objs_gt = np.vstack([objs_gt, np.zeros([self.seq_len,
                                                7])])[:self.seq_len]
        objs_obs = np.vstack([objs_obs, np.zeros([self.seq_len,
                                                  7])])[:self.seq_len]
        confidence = np.hstack([confidence,
                                np.zeros([self.seq_len])])[:self.seq_len]
        cam_loc = np.vstack([cam_loc, np.zeros([self.seq_len,
                                                3])])[:self.seq_len]
        cam_rot = np.vstack([cam_rot, np.zeros([self.seq_len,
                                                3])])[:self.seq_len]
        cen_gt = np.vstack([cen_gt, np.zeros([self.seq_len,
                                              2])])[:self.seq_len]
        dim_gt = np.vstack([dim_gt, np.zeros([self.seq_len,
                                              3])])[:self.seq_len]
        depth_gt = np.hstack([depth_gt,
                              np.zeros([self.seq_len])])[:self.seq_len]
        alpha_gt = np.hstack([alpha_gt,
                              np.zeros([self.seq_len])])[:self.seq_len]
        yaw_gt = np.hstack([yaw_gt, np.zeros([self.seq_len])])[:self.seq_len]
        yaw_axis_gt = np.vstack([yaw_axis_gt,
                                 np.zeros([self.seq_len, 2])])[:self.seq_len]
        cen_pd = np.vstack([cen_random,
                            np.zeros([self.seq_len, 2])])[:self.seq_len]
        dim_pd = np.vstack([dim_random,
                            np.zeros([self.seq_len, 3])])[:self.seq_len]
        depth_pd = np.hstack([depth_random,
                              np.zeros([self.seq_len])])[:self.seq_len]
        alpha_pd = np.hstack([alpha_random,
                              np.zeros([self.seq_len])])[:self.seq_len]
        yaw_pd = np.hstack([yaw_random,
                            np.zeros([self.seq_len])])[:self.seq_len]
        yaw_axis_pd = np.vstack([yaw_axis_pd,
                                 np.zeros([self.seq_len, 2])])[:self.seq_len]

        # Torch tensors
        traj_out = {
            'obj_gt': torch.from_numpy(objs_gt).float(),
            'obj_obs': torch.from_numpy(objs_obs).float(),
            'depth_gt': torch.from_numpy(depth_gt).float(),
            'alpha_gt': torch.from_numpy(alpha_gt).float(),
            'yaw_gt': torch.from_numpy(yaw_gt).float(),
            'yaw_axis_gt': torch.from_numpy(yaw_axis_gt).float(),
            'dim_gt': torch.from_numpy(dim_gt).float(),
            'cen_gt': torch.from_numpy(cen_gt).float(),
            'depth_pd': torch.from_numpy(depth_pd).float(),
            'alpha_pd': torch.from_numpy(alpha_pd).float(),
            'yaw_pd': torch.from_numpy(yaw_pd).float(),
            'yaw_axis_pd': torch.from_numpy(yaw_axis_pd).float(),
            'dim_pd': torch.from_numpy(dim_pd).float(),
            'cen_pd': torch.from_numpy(cen_pd).float(),
            'cam_rot': torch.from_numpy(cam_rot).float(),
            'cam_loc': torch.from_numpy(cam_loc).float(),
            'confidence': torch.from_numpy(confidence).float(),
            'valid_mask': torch.from_numpy(valid_mask).float()
        }
        return traj_out

    def convert_tracking_gt(self, sequence_data):
        tracking_dict = {}
        for fr_idx, frame in sequence_data['frames'].items():
            for obj_gt in frame['annotations']:
                tid = obj_gt['track_id']
                # If not ignore
                # Get rois, feature, depth, depth_gt, cam_rot, cam_trans
                tid_data = {
                    'depth_gt': obj_gt['location'][2],
                    'alpha_gt': obj_gt['alpha'],
                    'yaw_gt': obj_gt['yaw'],
                    'dim_gt': obj_gt['dimension'],
                    'center_gt': obj_gt['box_center'],
                    'loc_gt': obj_gt['location'],
                    'cam_calib': frame['cam_calib'],
                    'cam_rot': frame['cam_rot'],
                    'cam_loc': frame['cam_loc'],
                    'fr_idx': fr_idx
                }

                if tid not in tracking_dict:
                    tracking_dict[tid] = [tid_data.copy()]
                else:
                    tracking_dict[tid].append(tid_data.copy())

        return tracking_dict

    def convert_tracking_pred(self, sequence_data, sequence_result):
        tracking_dict = {}
        width = sequence_data['width']
        height = sequence_data['height']
        print(width, height)
        for fr_idx, frame_gt in sequence_data['frames'].items():
            frame_pd = sequence_result['frames'][fr_idx]
            obj_boxes = np.array(
                [obj_pd['box'] for obj_pd in frame_pd['annotations']])
            if len(obj_boxes):
                obj_boxes /= np.array([[width, height, width, height]])

            for obj_gt in frame_gt['annotations']:
                tid = obj_gt['track_id']
                # If not ignore
                # Get rois, feature, depth, depth_gt, cam_rot, cam_trans

                if len(obj_boxes):
                    _, box_idx, valid = tu.matching(
                        np.array(obj_gt['box']) /
                        np.array([[width, height, width, height]]),
                        obj_boxes,
                        thres=0.85)
                    if np.any(valid):
                        obj_pd = frame_pd['annotations'][box_idx.item()]
                    else:
                        obj_pd = obj_gt
                else:
                    obj_pd = obj_gt

                tid_data = {
                    'depth_gt': obj_gt['location'][2],
                    'alpha_gt': obj_gt['alpha'],
                    'yaw_gt': obj_gt['yaw'],
                    'dim_gt': obj_gt['dimension'],
                    'center_gt': obj_gt['box_center'],
                    'loc_gt': obj_gt['location'],
                    'depth_pd': obj_pd['location'][2],
                    'alpha_pd': obj_pd['alpha'],
                    'yaw_pd': obj_pd['yaw'],
                    'dim_pd': obj_pd['dimension'],
                    'center_pd': obj_pd['box_center'],
                    'loc_pd': obj_pd['location'],
                    'confidence_pd': obj_pd['confidence'],
                    'cam_calib': frame_gt['cam_calib'],
                    'cam_rot': frame_gt['cam_rot'],
                    'cam_loc': frame_gt['cam_loc'],
                    'fr_idx': fr_idx
                }

                if tid not in tracking_dict:
                    tracking_dict[tid] = [tid_data.copy()]
                else:
                    tracking_dict[tid].append(tid_data.copy())

        return tracking_dict

    def sample(self, data):
        datakey = []
        for key in list(data):
            if len(data[key]) > self.min_seq_len:
                datakey.append(key)
        return datakey


class MotionTrainer():

    def __init__(self, args):
        self.input_gt_path = [args.input_gt_path]
        self.input_pd_path = [args.input_pd_path]
        self.num_input_data = len(self.input_gt_path)
        self.ckpt_path = args.ckpt_path.format(args.session, args.set,
                                               args.num_epochs)
        self.cache_name = args.cache_name

        self.set = args.set
        self.phase = args.phase
        self.lstm_model = get_lstm_model(args.lstm_model_name)
        self.tracker_model = get_tracker(args.tracker_model_name)
        self.lstm_model_name = args.lstm_model_name
        self.tracker_model_name = args.tracker_model_name
        self.session = args.session
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.num_workers = args.num_workers

        self.resume = args.resume
        self.show_freq = args.show_freq
        self.is_verbose = args.is_verbose
        self.is_plot = args.is_plot
        self.is_train = args.phase == 'train'

        self.model = None
        self.num_seq = args.num_seq
        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.loc_dim = args.loc_dim

        self.data_loader = None
        self.train_loader = None
        self.val_loader = None
        self.seq_len = args.seq_len
        self.min_seq_len = args.min_seq_len
        self.max_depth = args.max_depth
        self.depth_scale = args.depth_scale

        self.optimizer = None
        self.init_lr = args.init_lr
        self.lr = args.init_lr
        self.step_ratio = args.step_ratio
        self.lr_adjust = args.lr_adjust
        self.lr_step = args.lr_step
        self.depth_weight = args.depth_weight
        self.weight_decay = args.weight_decay
        self.dropout = 0.1 if self.is_train else 0.0

    def loop_epoch(self):

        self._init_model()

        # Start epoch iterations
        for epoch in range(self.start_epoch, self.num_epochs + 1):

            if self.is_train:
                self.model.train()
                self.lr = nu.adjust_learning_rate(self.optimizer, epoch,
                                                  self.init_lr,
                                                  self.step_ratio,
                                                  self.lr_step, self.lr_adjust)

                self._init_dataset(is_train=True)
                self._loop_sequence(epoch, is_train=True)

                # Save
                if epoch % min(10, self.num_epochs) == 0:
                    torch.save(
                        {
                            'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'session': self.session,
                            'optimizer': self.optimizer.state_dict(),
                        }, self.ckpt_path)

            self.model.eval()
            with torch.no_grad():
                self._init_dataset(is_train=False)
                self._loop_sequence(epoch, is_train=False)

    def _loop_sequence(self, epoch: int, is_train: bool = True):

        losses = {
            'total_losses': tu.AverageMeter(),
            'pred_losses': tu.AverageMeter(),
            'refine_losses': tu.AverageMeter(),
            'linear_losses': tu.AverageMeter()
        }
        losses_kf = {
            'total_losses': tu.AverageMeter(),
            'pred_losses': tu.AverageMeter(),
            'refine_losses': tu.AverageMeter(),
            'linear_losses': tu.AverageMeter()
        }

        for iters, traj_out in enumerate(
                tqdm(self.data_loader, total=len(self.data_loader))):

            (obj_obs, obj_gt, loc_preds, loc_refines, loc_preds_kf,
             loc_refines_kf, confidence, valid_mask,
             cam_loc) = self._run_engine(traj_out)

            verbose("=" * 20, self.is_verbose)
            total_loss = self._loss_term(obj_obs, obj_gt, loc_preds,
                                         loc_refines, confidence, losses,
                                         valid_mask, epoch, iters,
                                         len(self.data_loader), cam_loc,
                                         self.lstm_model_name, is_train)
            _ = self._loss_term(obj_obs, obj_gt, loc_preds_kf, loc_refines_kf,
                                confidence, losses_kf, valid_mask, epoch,
                                iters, len(self.data_loader), cam_loc,
                                self.tracker_model_name, is_train)

            def closure():
                # Clear the states of model parameters each time
                self.optimizer.zero_grad()

                # BP loss
                total_loss.backward()

                # Clip if the gradients explode
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

                return total_loss

            if is_train:
                self.optimizer.step(closure)

    def _run_engine(self, trajs: torch.Tensor):

        # Initial
        cam_loc = trajs['cam_loc'].to(self.device)
        obj_gt = trajs['obj_gt'].to(self.device)
        obj_obs = trajs['obj_obs'].to(self.device)
        confidence = trajs['confidence'].to(self.device)
        valid_mask = trajs['valid_mask'].to(self.device)

        # batch x len x loc_dim
        obj_gt[..., :3] -= cam_loc[:, 0:1]
        obj_obs[..., :3] -= cam_loc[:, 0:1]

        loc_preds = []
        loc_refines = []
        loc_preds_kf = []
        loc_refines_kf = []

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        hidden_predict = self.model.init_hidden(self.device)  # None
        hidden_refine = self.model.init_hidden(self.device)  # None

        # Generate a history of location
        vel_history = obj_obs.new_zeros(self.num_seq, obj_obs.shape[0],
                                        self.loc_dim)

        # Starting condition
        prev_refine = obj_obs[:, 0, :self.loc_dim]
        loc_pred = obj_obs[:, 1, :self.loc_dim]

        with torch.no_grad():
            trks = [
                self.tracker_model(self.device, self.model, _box, _conf)
                if self.tracker_model_name == 'LSTM3DTracker' else
                self.tracker_model(_box, _conf)
                for _box, _conf in zip(obj_obs[:, 0].cpu().numpy(),
                                       confidence.cpu().numpy())
            ]

        for i in range(1, valid_mask.shape[1]):
            # LSTM
            loc_pred[:, 3:4] = fix_alpha(loc_pred[:, 3:4])

            for batch_id in range(obj_obs.shape[0]):
                curr_yaw = fix_alpha(obj_obs[batch_id, i, 3:4])
                if np.pi / 2.0 < abs(
                        curr_yaw - loc_pred[batch_id, 3:4]) < np.pi * 3 / 2.0:
                    loc_pred[batch_id, 3:4] += np.pi
                    if loc_pred[batch_id, 3:4] > np.pi:
                        loc_pred[batch_id, 3:4] -= np.pi * 2
                    if loc_pred[batch_id, 3:4] < -np.pi:
                        loc_pred[batch_id, 3:4] += np.pi * 2

                # now the angle is acute: < 90 or > 270,
                # convert the case of > 270 to < 90
                if abs(curr_yaw - loc_pred[batch_id, 3:4]) >= np.pi * 3 / 2.0:
                    if curr_yaw > 0:
                        loc_pred[batch_id, 3:4] += np.pi * 2
                    else:
                        loc_pred[batch_id, 3:4] -= np.pi * 2

            loc_refine, hidden_refine = self.model.refine(
                loc_pred.detach().clone(), obj_obs[:, i, :self.loc_dim],
                prev_refine.detach().clone(), confidence[:, i, None],
                hidden_refine)
            loc_refine[:, 3:4] = fix_alpha(loc_refine[:, 3:4])

            if i == 1:
                vel_history = torch.cat(
                    [(loc_refine - prev_refine).unsqueeze(0)] * self.num_seq)
            else:
                vel_history = torch.cat(
                    [vel_history[1:], (loc_refine - prev_refine).unsqueeze(0)],
                    dim=0)
            prev_refine = loc_refine

            loc_pred, hidden_predict = self.model.predict(
                vel_history,
                loc_refine.detach().clone(), hidden_predict)
            loc_pred[:, 3:4] = fix_alpha(loc_pred[:, 3:4])

            # KF3D
            with torch.no_grad():
                for trk_idx, trk in enumerate(trks):
                    if i == 1:
                        trk.predict(update_state=False)

                    trk.update(obj_obs[trk_idx, i].cpu().numpy(),
                               confidence[trk_idx, i].cpu().numpy())

                loc_refine_kf = loc_refine.new(
                    np.vstack([trk.get_state()[:self.loc_dim]
                               for trk in trks]))

                loc_pred_kf = loc_pred.new(
                    np.vstack([
                        trk.predict().squeeze()[:self.loc_dim] for trk in trks
                    ]))

            # Predict residual of depth
            loc_preds.append(loc_pred)
            loc_refines.append(loc_refine)
            loc_preds_kf.append(loc_pred_kf)
            loc_refines_kf.append(loc_refine_kf)

        return (obj_obs, obj_gt, loc_preds, loc_refines, loc_preds_kf,
                loc_refines_kf, confidence, valid_mask, cam_loc)

    def _loss_term(self, loc_obs, loc_gt, loc_preds, loc_refines, confidence,
                   losses, valid_mask, epoch, iters, num_iters, cam_loc,
                   method: str, is_train: bool) -> torch.Tensor:

        loc_refines = torch.cat(
            loc_refines, dim=1).view(valid_mask.shape[0], -1, self.loc_dim)
        loc_preds = torch.cat(
            loc_preds, dim=1).view(valid_mask.shape[0], -1, self.loc_dim)

        if self.loc_dim > 3:
            loc_refines[
                ..., 3] = (loc_refines[..., 3] + np.pi) % (2 * np.pi) - np.pi
            loc_preds[...,
                      3] = (loc_preds[..., 3] + np.pi) % (2 * np.pi) - np.pi

        loc_refine_mask = loc_refines[:, :] * valid_mask[:, 1:, None]
        loc_pred_mask = loc_preds[:, :-1] * valid_mask[:, 2:, None]
        loc_gt_mask = loc_gt[:, :, :self.loc_dim] * valid_mask[:, :, None]
        loc_obs_mask = loc_obs[:, :, :self.loc_dim] * valid_mask[:, :, None]

        # Normalize yaw angle
        loc_refine_mask[loc_refine_mask[:, :, 3] < 0][:, 3] += 2 * np.pi
        loc_pred_mask[loc_pred_mask[:, :, 3] < 0][:, 3] += 2 * np.pi

        # Cost functions
        refine_loss = F.smooth_l1_loss(
            loc_refine_mask, loc_gt_mask[:, 1:], reduction='sum') / torch.sum(
                valid_mask[:, 1:])
        pred_loss = F.smooth_l1_loss(
            loc_pred_mask, loc_gt_mask[:, 2:], reduction='sum') / torch.sum(
                valid_mask[:, 2:])
        linear_loss = nu.linear_motion_loss(loc_refine_mask, valid_mask[:, 1:])
        linear_loss += nu.linear_motion_loss(loc_pred_mask, valid_mask[:, 2:])

        verbose(method, self.is_verbose)
        verbose(
            f"Ref: {torch.mean(loc_refine_mask - loc_gt_mask[:, 1:], dim=0).detach().cpu().numpy()}",
            self.is_verbose)
        verbose(
            f"Prd: {torch.mean(loc_pred_mask - loc_gt_mask[:, 2:], dim=0).detach().cpu().numpy()}",
            self.is_verbose)

        total_loss: torch.Tensor = (
            self.depth_weight * (refine_loss + pred_loss) +
            (1.0 - self.depth_weight) * linear_loss)

        # Updates
        losses['total_losses'].update(total_loss.data.cpu().numpy().item(),
                                      int(torch.sum(valid_mask)))
        losses['pred_losses'].update(pred_loss.data.cpu().numpy().item(),
                                     int(torch.sum(valid_mask)))
        losses['refine_losses'].update(refine_loss.data.cpu().numpy().item(),
                                       int(torch.sum(valid_mask)))
        losses['linear_losses'].update(linear_loss.data.cpu().numpy().item(),
                                       int(torch.sum(valid_mask)))

        # Verbose
        if iters % min(num_iters - 1, self.show_freq) == 0 and iters != 0:

            phase = 'Train' if is_train else 'Val'

            status_msg = (f'[{self.set.upper()} - {self.session} | '
                          f'{phase} - {method}]\t'
                          f'[Epoch: {epoch}/{self.num_epochs} | '
                          f'Iters: {iters}/{len(self.data_loader)}')
            if is_train:
                status_msg += f' | LR: {self.lr:.6f}]\t'
            else:
                status_msg += ']\t'

            verbose(
                f'{status_msg}'
                '[Total Loss {loss.val:2.2f} ({loss.avg:2.2f}) | '
                'P-Loss {pred.val:2.2f} ({pred.avg:2.2f}) | '
                'R-Loss {refine.val:2.2f} ({refine.avg:2.2f}) | '
                'S-Loss {smooth.val:2.2f} ({smooth.avg:2.2f})]'.format(
                    loss=losses['total_losses'],
                    pred=losses['pred_losses'],
                    refine=losses['refine_losses'],
                    smooth=losses['linear_losses']), True)
            verbose(
                f"PD: {loc_pred_mask[0].cpu().data.numpy()}\n"
                f"OB: {loc_obs_mask[0].cpu().data.numpy()}\n"
                f"RF: {loc_refine_mask[0].cpu().data.numpy()}\n"
                f"GT: {loc_gt_mask[0].cpu().data.numpy()}\n"
                f"Conf: {confidence[0].cpu().data.numpy()}", self.is_verbose)

            if self.is_plot:
                plot_3D(
                    osp.dirname(self.cache_name),
                    f"{epoch}_{iters}_{phase}_{method}",
                    self.session,
                    cam_loc[0].cpu().data.numpy(),
                    loc_gt[0].cpu().data.numpy(),
                    predictions={
                        'Obs': loc_obs[0].cpu().data.numpy(),
                        'Prd': loc_preds[0].cpu().data.numpy(),
                        'Ref': loc_refines[0].cpu().data.numpy()
                    },
                    show_cam_loc=False)

        return total_loss

    def _init_model(self):
        self.model = self.lstm_model(
            self.batch_size,
            self.feature_dim,
            self.hidden_size,
            self.num_layers,
            self.loc_dim,
            dropout=self.dropout).to(self.device)

        if self.is_train:
            self.model.train()
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
                amsgrad=True)
            if self.resume:
                nu.load_checkpoint(
                    self.model,
                    self.ckpt_path,
                    optimizer=self.optimizer,
                    is_test=not self.is_train)
        else:
            self.model.eval()
            nu.load_checkpoint(
                self.model,
                self.ckpt_path,
                optimizer=self.optimizer,
                is_test=not self.is_train)

    def _init_dataset(self, is_train: bool = True):

        if is_train and self.train_loader is not None:
            verbose(f"TRAIN set with {len(self.train_loader)} trajectories",
                    True)
            self.data_loader = self.train_loader
        elif not is_train and self.val_loader is not None:
            verbose(f"VAL set with {len(self.val_loader)} trajectories", True)
            self.data_loader = self.val_loader
        else:
            # Data loading code
            dataset = SeqDataset(self.set, self.input_gt_path,
                                 self.input_pd_path, is_train, self.seq_len,
                                 self.min_seq_len, self.max_depth,
                                 self.depth_scale, self.cache_name)
            verbose(f"Generate dataset with {dataset.__len__()} trajectories ",
                    True)

            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=is_train,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True)

            if is_train:
                self.data_loader = data_loader
                self.train_loader = data_loader
            else:
                self.data_loader = data_loader
                self.val_loader = data_loader


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='RNN depth motion estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti', 'nuscenes', 'waymo'])
    parser.add_argument(
        'phase',
        choices=['train', 'test'],
        help='Which data split to use in testing')
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test', 'mini'],
        default='train',
        help='Which data split to use in testing')

    parser.add_argument(
        '--input_pd_path',
        dest='input_pd_path',
        help='path of input pred info for tracking',
        default='./data/KITTI/anns/tracking_train_output.json',
        type=str)
    parser.add_argument(
        '--input_gt_path',
        dest='input_gt_path',
        help='path of input gt info for tracking',
        default='./data/KITTI/anns/tracking_train.json',
        type=str)
    parser.add_argument(
        '--cache_name',
        dest='cache_name',
        help='path of cache file',
        default='./work_dirs/LSTM/kitti_train_full_traj.pkl',
        type=str)
    parser.add_argument(
        '--session',
        dest='session',
        help='session of tracking',
        default='batch10',
        type=str)
    parser.add_argument(
        '--ckpt_path',
        dest='ckpt_path',
        help='path of checkpoint file',
        default='./checkpoints/{}_{}_{:03d}_linear.pth',
        type=str)
    parser.add_argument(
        '--lstm_model_name',
        dest='lstm_model_name',
        help='Name of the LSTM model',
        default='LocLSTM',
        choices=LSTM_MODEL_ZOO.keys(),
        type=str)
    parser.add_argument(
        '--tracker_model_name',
        dest='tracker_model_name',
        help='Name of the LSTM model',
        default='LSTM3DTracker',
        choices=TRACKER_MODEL_ZOO.keys(),
        type=str)
    parser.add_argument(
        '--start_epoch',
        default=0,
        type=int,
        help='manual epoch number (useful on restarts)')
    parser.add_argument(
        '--seq_len',
        dest='seq_len',
        help='sequence length feed to model',
        default=10,
        type=int)
    parser.add_argument(
        '--min_seq_len',
        dest='min_seq_len',
        help='minimum available sequence length',
        default=10,
        type=int)
    parser.add_argument(
        '--depth_scale',
        dest='depth_scale',
        help='depth uncertainty in training (no uncertainty when value <= 0)',
        default=10,
        type=int)
    parser.add_argument(
        '--max_depth',
        dest='max_depth',
        help='maximum depth in training',
        default=100,
        type=int)
    parser.add_argument(
        '--min_depth',
        dest='min_depth',
        help='minimum depth in training',
        default=0,
        type=int)
    parser.add_argument(
        '--show_freq',
        dest='show_freq',
        help='verbose frequence',
        default=100,
        type=int)
    parser.add_argument(
        '--feature_dim',
        dest='feature_dim',
        help='feature dimension feed into model',
        default=64,
        type=int)
    parser.add_argument(
        '--loc_dim',
        dest='loc_dim',
        help='output dimension, we model depth here',
        default=3,
        type=int)
    parser.add_argument(
        '--hidden_size',
        dest='hidden_size',
        help='hidden size of LSTM',
        default=128,
        type=int)
    parser.add_argument(
        '--num_layers',
        dest='num_layers',
        help='number of layers of LSTM',
        default=2,
        type=int)
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='number of epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--num_seq',
        dest='num_seq',
        help='number of seq used in predicting next step',
        default=5,
        type=int)
    parser.add_argument(
        '--init_lr',
        default=5e-3,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        '--lr-adjust',
        help='learning rate adjust strategy',
        choices=['step'],
        default='step',
        type=str)
    parser.add_argument(
        '--lr-step', help='number of steps to decay lr', default=20, type=int)
    parser.add_argument(
        '--step-ratio', dest='step_ratio', default=0.5, type=float)
    parser.add_argument(
        '--depth_weight',
        dest='depth_weight',
        help='weight of depth and smooth loss',
        default=0.9,
        type=float)
    parser.add_argument(
        '--weight-decay',
        '--wd',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)')
    parser.add_argument(
        '-j',
        '--num_workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 16)')
    parser.add_argument(
        '-b',
        '--batch_size',
        default=10,
        type=int,
        help='the batch size on each gpu')
    parser.add_argument(
        '--is_plot',
        dest='is_plot',
        help='show prediction result',
        default=False,
        action='store_true')
    parser.add_argument(
        '--resume',
        dest='resume',
        help='resume model checkpoint',
        default=False,
        action='store_true')
    parser.add_argument(
        '--verbose',
        dest='is_verbose',
        help='verbose',
        default=False,
        action='store_true')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(args)

    return args


def main():
    """Train an object motion LSTM from input location sequences
    """
    args = parse_args()
    if args.phase == 'test':
        assert args.batch_size == 1, "Inference with batch size 1 only"
        args.start_epoch = args.num_epochs
    torch.set_num_threads(multiprocessing.cpu_count())
    cudnn.benchmark = True
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    motion_trainer = MotionTrainer(args)
    motion_trainer.loop_epoch()


if __name__ == '__main__':
    print(f"Torch version: {torch.__version__}")
    main()
