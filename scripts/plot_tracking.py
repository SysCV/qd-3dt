# coding: utf-8
import os
import sys
import cv2
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from pyquaternion import Quaternion

from scripts import kitti_utils as ku
from scripts import tracking_utils as tu
from scripts import plot_utils as pu

from scripts.object_ap_eval.coco_format import read_file as coco_rf

print(f"Using {matplotlib.get_backend()} as matplotlib backend")

cat_mapping = {
    'kitti': ['Car', 'Pedestrian', 'Cyclist'],
    'gta': ['Car'],
    'nuscenes':
    ['Bicycle', 'Motorcycle', 'Pedestrian', 'Bus', 'Car', 'Trailer', 'Truck'],
    'waymo': ['Car', 'Pedestrian', 'Cyclist'],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Monocular 3D Tracking Visualizer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', help='dataset name')
    parser.add_argument('gt_folder', help='groundtruth label folder')
    parser.add_argument('--res_folder', help='result txt folder')
    parser.add_argument(
        '--fps', default=7, type=int, help='fps to store video')
    parser.add_argument(
        '--is_save',
        default=False,
        action='store_true',
        help='whether to merge two video or not')
    parser.add_argument(
        '--is_merge',
        default=False,
        action='store_true',
        help='whether to merge two video or not')
    parser.add_argument(
        '--is_gt',
        default=False,
        action='store_true',
        help='whether to plot annotation or not')
    parser.add_argument(
        '--is_remove', default=False, action='store_true', help='remove video')
    parser.add_argument(
        '--draw_3d', default=False, action='store_true', help='draw 3D box')
    parser.add_argument(
        '--draw_2d', default=False, action='store_true', help='draw 2D box')
    parser.add_argument(
        '--draw_bev',
        default=False,
        action='store_true',
        help='draw Birds eye view')
    parser.add_argument(
        '--draw_traj',
        default=False,
        action='store_true',
        help='draw 3D center trajectory')
    parser.add_argument(
        '--draw_tid',
        default=False,
        action='store_true',
        help='draw 3D center trajectory')
    parser.add_argument(
        '--align_gt',
        default=False,
        action='store_true',
        help='whether to align tid with gt or not')
    args = parser.parse_args()

    print(' '.join(sys.argv))

    if not args.is_gt and (args.is_save or args.is_merge or args.is_remove):
        assert args.res_folder is not None

    return args


def merge_vid(vidname1, vidname2, outputname):
    assert os.path.isfile(vidname1)
    assert os.path.isfile(vidname2)
    print(f"Vertically stack {vidname1} and {vidname2}, save as {outputname}")
    os.makedirs(os.path.dirname(outputname), exist_ok=True)

    # Get input video capture
    cap1 = cv2.VideoCapture(vidname1)
    cap2 = cv2.VideoCapture(vidname2)

    # Default resolutions of the frame are obtained.The default resolutions
    # are system dependent.
    # We convert the resolutions from float to integer.
    # https://docs.opencv.org/2.4/modules/highgui/doc
    # /reading_and_writing_images_and_video.html#videocapture-get
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    fps1 = cap1.get(5)
    FOURCC = int(cap1.get(6))
    num_frames = int(cap1.get(7))

    # frame_width2 = int(cap2.get(3))
    frame_height2 = int(cap2.get(4))
    fps2 = cap2.get(5)
    num_frames2 = int(cap2.get(7))

    print(fps1, fps2)
    if fps1 > fps2:
        fps = fps1
    else:
        fps = fps2

    # assert frame_height == frame_height2, \
    #     f"Height of frames are not equal. {frame_height} vs. {frame_height2}"
    assert num_frames == num_frames2, \
        f"Number of frames are not equal. {num_frames} vs. {num_frames2}"

    # Set output videowriter
    vidsize = (frame_width + frame_height, frame_height)
    out = cv2.VideoWriter(outputname, FOURCC, fps, vidsize)

    # Loop over and save
    print(f"Total {num_frames} frames. Now saving...")
    idx = 0
    while (cap1.isOpened() and cap2.isOpened() and idx < num_frames):
        ret1 = ret2 = False
        frame1 = frame2 = None
        if idx % (fps / fps1) == 0.0:
            # print(idx, fps/fps2, "1")
            ret1, frame1 = cap1.read()
        if idx % (fps / fps2) == 0.0:
            # print(idx, fps/fps1, "2")
            ret2, frame2 = cap2.read()
            if frame_height != frame_height2:
                frame2 = cv2.resize(frame2, (frame_height, frame_height))
        # print(ret1, ret2)
        if ret1 and ret2:
            out_frame = np.hstack([frame1, frame2])
            out.write(out_frame)
        idx += 1

    out.release()
    cap1.release()
    cap2.release()
    print(f'{outputname} Done!')


class Visualizer():

    def __init__(self,
                 dataset: str,
                 res_folder: str,
                 fps: float = 7.0,
                 draw_bev: bool = True,
                 draw_2d: bool = False,
                 draw_3d: bool = True,
                 draw_traj: bool = True,
                 draw_tid: bool = True,
                 is_save: bool = True,
                 is_merge: bool = True,
                 is_remove: bool = True,
                 is_gt: bool = True,
                 align_gt: bool = True):
        # Parameters
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE: float = 1.0
        self.FONT_THICKNESS: int = 1
        self.FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
        self.FOCAL_LENGTH = None
        self.fps: float = fps
        self.resW: int = None
        self.resH: int = None

        np.random.seed(777)

        # Create canvas
        sns.set(style="darkgrid")
        self.fig_size: int = 10
        self.dpi: int = 100
        self.bev_size: int = self.fig_size * self.dpi
        self.fig, self.ax = plt.subplots(
            figsize=(self.fig_size, self.fig_size), dpi=self.dpi)
        self.x_min: int = -55
        self.x_max: int = 55
        self.y_min: int = 0
        self.y_max: int = 100
        self.interval: int = 10
        self.num_hist: int = 10

        self.dataset = dataset
        if res_folder is None:
            self.res_folder = f'work_dirs/GT_Visualization/{dataset}'
        else:
            self.res_folder = os.path.dirname(res_folder)
        self.draw_bev: bool = draw_bev
        self.draw_2d: bool = draw_2d
        self.draw_3d: bool = draw_3d
        self.draw_traj: bool = draw_traj
        self.draw_tid: bool = draw_tid
        self.is_save: bool = is_save
        self.is_merge: bool = is_merge
        self.is_remove: bool = is_remove
        self.is_gt: bool = is_gt
        self.align_gt: bool = align_gt and is_gt

        # Variables
        self.trk_vid_name: str = None
        self.bev_vid_name: str = None

    @staticmethod
    def get_3d_info(anno, cam_calib, cam_pose):
        h, w, l = anno['dimension']
        depth = anno['location'][2]
        alpha = anno['alpha']
        xc, yc = anno['box_center']
        obj_class = anno['obj_type']

        points_cam = tu.imagetocamera(
            np.array([[xc, yc]]), np.array([depth]), cam_calib)

        bev_center = points_cam[0, [0, 2]]
        yaw = tu.alpha2rot_y(alpha, bev_center[0], bev_center[1])  # rad
        quat_yaw = Quaternion(axis=[0, 1, 0], radians=yaw)
        quat_cam_rot = Quaternion(matrix=cam_pose.rotation)
        quat_yaw_world = quat_cam_rot * quat_yaw

        box3d = tu.computeboxes([yaw], (h, w, l), points_cam)
        points_world = tu.cameratoworld(points_cam, cam_pose)

        output = {
            'center': bev_center,
            'loc_cam': points_cam,
            'loc_world': points_world,
            'yaw': yaw,
            'yaw_world_quat': quat_yaw_world,
            'box3d': box3d,
            'class': obj_class
        }
        return output

    @staticmethod
    def draw_3d_traj(frame,
                     points_hist,
                     cam_calib,
                     cam_pose,
                     line_color=(0, 255, 0)):

        # Plot center history
        for index, wt in enumerate(points_hist):
            ct = tu.worldtocamera(wt, cam_pose)
            pt = tu.cameratoimage(ct, cam_calib)
            rgba = line_color + tuple(
                [int(max(float(index) / len(points_hist), 0.5) * 255)])
            cv2.circle(
                frame, (int(pt[0, 0]), int(pt[0, 1])), 3, rgba, thickness=-1)

        return frame

    def draw_corner_info(self, frame, x1, y1, info_str, line_color):
        (test_width,
         text_height), baseline = cv2.getTextSize(info_str, self.FONT,
                                                  self.FONT_SCALE * 0.5,
                                                  self.FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - text_height),
                      (x1 + test_width, y1 + baseline), line_color, cv2.FILLED)
        cv2.putText(frame, info_str, (x1, y1), self.FONT,
                    self.FONT_SCALE * 0.5, (0, 0, 0), self.FONT_THICKNESS,
                    cv2.LINE_AA)
        return frame

    def draw_bev_canvas(self):
        # Set x, y limit and mark border
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.set_xlim(self.x_min - 1, self.x_max + 1)
        self.ax.set_ylim(self.y_min - 1, self.y_max + 1)
        self.ax.tick_params(axis='both', labelbottom=False, labelleft=False)
        self.ax.xaxis.set_minor_locator(MultipleLocator(self.interval))
        self.ax.yaxis.set_minor_locator(MultipleLocator(self.interval))

        for radius in range(self.y_max, -1, -self.interval):
            # Mark all around sector
            self.ax.add_patch(
                mpatches.Wedge(
                    center=[0, 0],
                    alpha=0.1,
                    aa=True,
                    r=radius,
                    theta1=-180,
                    theta2=180,
                    fc="black"))

            # Mark range
            if radius / np.sqrt(2) + 8 < self.x_max:
                self.ax.text(
                    radius / np.sqrt(2) + 3,
                    radius / np.sqrt(2) - 5,
                    f'{radius}m',
                    rotation=-45,
                    color='darkblue',
                    fontsize='xx-large')

        # Mark visible sector
        self.ax.add_patch(
            mpatches.Wedge(
                center=[0, 0],
                alpha=0.1,
                aa=True,
                r=self.y_max,
                theta1=45,
                theta2=135,
                fc="cyan"))

        # Mark ego-vehicle
        self.ax.arrow(0, 0, 0, 3, color='black', width=0.5, overhang=0.3)

    def draw_2d_bbox(self,
                     frame,
                     box,
                     line_color: tuple = (0, 255, 0),
                     line_width: int = 3,
                     corner_info: str = None):

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), line_color,
                      line_width)

        if corner_info is not None:
            x1 = int(box[0])
            y1 = int(box[3])

            frame = self.draw_corner_info(frame, x1, y1, corner_info,
                                          line_color)

        return frame

    def draw_3d_bbox(self,
                     frame,
                     points_camera,
                     cam_calib,
                     cam_pose,
                     cam_near_clip: float = 0.15,
                     line_color: tuple = (0, 255, 0),
                     line_width: int = 3,
                     corner_info: str = None):
        projpoints = tu.get_3d_bbox_vertex(cam_calib, cam_pose, points_camera,
                                           cam_near_clip)

        for p1, p2 in projpoints:
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     line_color, line_width)

        if corner_info is not None:
            is_before = False
            cp1 = tu.cameratoimage(points_camera[0:1], cam_calib)[0]

            if cp1 is not None:
                is_before = tu.is_before_clip_plane_camera(
                    points_camera[0:1], cam_near_clip)[0]

            if is_before:
                x1 = int(cp1[0])
                y1 = int(cp1[1])

                frame = self.draw_corner_info(frame, x1, y1, corner_info,
                                              line_color)

        return frame

    def plot_3D_box(self, n_seq_pd: str, pd_seq: dict, gt_seq: dict):
        id_to_color = {}
        cmap = pu.RandomColor(len(gt_seq['frames']))

        # Variables
        subfolder = 'shows_3D' if self.draw_3d else 'shows_2D'
        self.trk_vid_name = os.path.join(self.res_folder, subfolder,
                                         f'{n_seq_pd}_tracking.mp4')
        self.bev_vid_name = os.path.join(self.res_folder, 'shows_BEV',
                                         f'{n_seq_pd}_birdsview.mp4')

        print(f"Trk: {self.trk_vid_name}")
        print(f"BEV: {self.bev_vid_name}")

        self.resH = gt_seq['height']
        self.resW = gt_seq['width']
        rawimg = None
        vid_trk = None
        vid_bev = None

        # set output video
        if self.is_save:
            os.makedirs(os.path.dirname(self.trk_vid_name), exist_ok=True)
            os.makedirs(os.path.dirname(self.bev_vid_name), exist_ok=True)
            vid_trk = cv2.VideoWriter(self.trk_vid_name, self.FOURCC, self.fps,
                                      (self.resW, self.resH))
            vid_bev = cv2.VideoWriter(self.bev_vid_name, self.FOURCC, self.fps,
                                      (self.bev_size, self.bev_size))

        max_frames = max(len(pd_seq['frames']), len(gt_seq['frames']))
        print(f"Seq ID: {n_seq_pd}\n"
              f"Total frames: {max_frames}\n"
              f"PD frames: {len(pd_seq['frames'])}\n"
              f"GT frames: {len(gt_seq['frames'])}")

        loc_world_hist_gt = {}
        loc_world_hist_pd = {}

        for n_frame in range(max_frames):

            self.FOCAL_LENGTH = gt_seq['frames'][n_frame]['cam_calib'][0][0]

            gt_objects = gt_seq['frames'].get(n_frame, {'annotations': []})
            pd_objects = pd_seq['frames'].get(n_frame, {'annotations': []})
            gt_annos = {}
            pd_annos = {}
            boxes_2d_pd = None

            if n_frame % 100 == 0:
                print(f"Seq {n_seq_pd}, Frame {n_frame}")

            # Get objects
            if self.draw_3d or self.draw_2d:
                rawimg = cv2.imread(gt_objects['im_path'])
                (test_width, text_height), baseline = cv2.getTextSize(
                    str(n_frame), self.FONT, self.FONT_SCALE,
                    self.FONT_THICKNESS * 2)
                cv2.rectangle(rawimg, (0, 0),
                              (test_width, text_height + baseline),
                              (255, 255, 255), -1)
                cv2.putText(rawimg, str(n_frame),
                            (0, text_height + baseline // 2), self.FONT,
                            self.FONT_SCALE, (0, 0, 0),
                            self.FONT_THICKNESS * 2, cv2.LINE_AA)

            cam_coords = np.array(gt_objects['cam_loc'])
            cam_rotation = np.array(gt_objects['cam_rot'])
            cam_calib = np.array(gt_objects['cam_calib'])
            cam_pose = ku.Pose(cam_coords, cam_rotation)

            if self.align_gt:
                boxes_2d_pd = [
                    hypo['box'] for hypo in pd_objects['annotations']
                ]

            if len(gt_objects['annotations']) > 0 and self.is_gt:
                gt_annos = sorted(
                    gt_objects['annotations'],
                    key=lambda x: x['location'][2],
                    reverse=True)

            if len(pd_objects['annotations']) > 0:
                pd_annos = sorted(
                    pd_objects['annotations'],
                    key=lambda x: x['location'][2],
                    reverse=True)

            for anno in gt_annos:

                tid_gt_str = f"{anno['track_id']}GT"
                tid_gt = anno['track_id']
                box_gt = np.array(anno['box']).astype(int)
                _, w_gt, l_gt = anno['dimension']
                anno_dict = self.get_3d_info(anno, cam_calib, cam_pose)
                center_gt = anno_dict['center']
                loc_world_gt = anno_dict['loc_world']
                yaw_gt = anno_dict['yaw']
                yaw_world_gt = anno_dict['yaw_world_quat']
                box3d_gt = anno_dict['box3d']
                obj_class = anno_dict['class']
                if tid_gt not in loc_world_hist_gt:
                    loc_world_hist_gt[tid_gt] = {
                        'loc': [loc_world_gt],
                        'yaw': [yaw_world_gt]
                    }
                elif len(loc_world_hist_gt[tid_gt]['loc']) > self.num_hist:
                    loc_world_hist_gt[tid_gt]['loc'] = loc_world_hist_gt[
                        tid_gt]['loc'][1:] + [loc_world_gt]
                    loc_world_hist_gt[tid_gt]['yaw'] = loc_world_hist_gt[
                        tid_gt]['yaw'][1:] + [yaw_world_gt]
                else:
                    loc_world_hist_gt[tid_gt]['loc'].append(loc_world_gt)
                    loc_world_hist_gt[tid_gt]['yaw'].append(yaw_world_gt)

                # Match gt and pd
                if self.align_gt:
                    _, idx, valid = tu.matching(
                        np.array(anno['box']).reshape(-1, 4),
                        np.array(boxes_2d_pd).reshape(-1, 4), 0.1)
                    if valid is not None and valid.item():
                        pd_objects['annotations'][
                            idx[0]]['match_id_str'] = tid_gt_str
                        pd_objects['annotations'][idx[0]]['match_id'] = tid_gt
                        pd_objects['annotations'][idx[0]]['match'] = True

                # Get box color
                # color is in BGR format (for cv2), color[:-1] in RGB format
                # (for plt)
                if tid_gt_str not in list(id_to_color):
                    id_to_color[tid_gt_str] = cmap.get_random_color(scale=255)
                color = id_to_color[tid_gt_str]

                if self.draw_tid:
                    info_str = f"{obj_class}{tid_gt_str}"
                else:
                    info_str = f"{obj_class}"

                # Make rectangle
                if self.draw_3d:
                    rawimg = self.draw_3d_bbox(
                        rawimg,
                        box3d_gt,
                        cam_calib,
                        cam_pose,
                        line_color=(color[0], color[1] * 0.7, color[2] * 0.7),
                        line_width=2,
                        corner_info=info_str)

                if self.draw_2d:
                    self.draw_2d_bbox(
                        rawimg,
                        box_gt,
                        line_color=(color[0], color[1] * 0.7, color[2] * 0.7),
                        line_width=3,
                        corner_info=info_str)

                if self.draw_traj:
                    # Draw trajectories
                    rawimg = self.draw_3d_traj(
                        rawimg,
                        loc_world_hist_gt[tid_gt]['loc'],
                        cam_calib,
                        cam_pose,
                        line_color=color)

                if self.draw_bev:
                    # Change BGR to RGB
                    color_bev = [c / 255.0 for c in color[::-1]]
                    center_hist_gt = tu.worldtocamera(
                        np.vstack(loc_world_hist_gt[tid_gt]['loc']),
                        cam_pose)[:, [0, 2]]
                    quat_cam_rot_t = Quaternion(matrix=cam_pose.rotation.T)
                    yaw_hist_gt = []
                    for quat_yaw_world_gt in loc_world_hist_gt[tid_gt]['yaw']:
                        rotation_cam = quat_cam_rot_t * quat_yaw_world_gt
                        vtrans = np.dot(rotation_cam.rotation_matrix,
                                        np.array([1, 0, 0]))
                        yaw_hist_gt.append(
                            -np.arctan2(vtrans[2], vtrans[0]).tolist())
                    yaw_hist_gt = np.vstack(yaw_hist_gt)
                    pu.plot_bev_obj(
                        self.ax,
                        center_gt,
                        center_hist_gt,
                        yaw_gt,
                        yaw_hist_gt,
                        l_gt,
                        w_gt,
                        color_bev,
                        'GT',
                        line_width=2)

            for hypo in pd_annos:

                # Get information of gt and pd
                if self.align_gt and hypo.get('match', False):
                    tid_pd_str = hypo['match_id_str']
                    tid_pd = hypo['match_id']
                else:
                    tid_pd_str = f"{hypo['track_id']}PD"
                    tid_pd = hypo['track_id']
                box_pd = np.array(hypo['box']).astype(int)
                _, w_pd, l_pd = hypo['dimension']
                hypo_dict = self.get_3d_info(hypo, cam_calib, cam_pose)
                center_pd = hypo_dict['center']
                loc_world_pd = hypo_dict['loc_world']
                yaw_pd = hypo_dict['yaw']
                yaw_world_pd = hypo_dict['yaw_world_quat']
                box3d_pd = hypo_dict['box3d']
                obj_class_pd = hypo_dict['class']
                if tid_pd not in loc_world_hist_pd:
                    loc_world_hist_pd[tid_pd] = {
                        'loc': [loc_world_pd],
                        'yaw': [yaw_world_pd]
                    }
                elif len(loc_world_hist_pd[tid_pd]['loc']) > self.num_hist:
                    loc_world_hist_pd[tid_pd]['loc'] = loc_world_hist_pd[
                        tid_pd]['loc'][1:] + [loc_world_pd]
                    loc_world_hist_pd[tid_pd]['yaw'] = loc_world_hist_pd[
                        tid_pd]['yaw'][1:] + [yaw_world_pd]
                else:
                    loc_world_hist_pd[tid_pd]['loc'].append(loc_world_pd)
                    loc_world_hist_pd[tid_pd]['yaw'].append(yaw_world_pd)

                # Get box color
                # color is in BGR format (for cv2), color[:-1] in RGB format
                # (for plt)
                if tid_pd_str not in list(id_to_color):
                    id_to_color[tid_pd_str] = cmap.get_random_color(scale=255)
                color = id_to_color[tid_pd_str]

                if self.draw_tid:
                    info_str = f"{obj_class_pd}{tid_pd}PD"
                else:
                    info_str = f"{obj_class_pd}"

                # Make rectangle
                if self.draw_3d:
                    # Make rectangle
                    rawimg = self.draw_3d_bbox(
                        rawimg,
                        box3d_pd,
                        cam_calib,
                        cam_pose,
                        line_color=color,
                        corner_info=info_str)

                if self.draw_2d:
                    self.draw_2d_bbox(
                        rawimg,
                        box_pd,
                        line_color=color,
                        line_width=3,
                        corner_info=info_str)

                if self.draw_traj:
                    # Draw trajectories
                    rawimg = self.draw_3d_traj(
                        rawimg,
                        loc_world_hist_pd[tid_pd]['loc'],
                        cam_calib,
                        cam_pose,
                        line_color=color)

                if self.draw_bev:
                    # Change BGR to RGB
                    color_bev = [c / 255.0 for c in color[::-1]]
                    center_hist_pd = tu.worldtocamera(
                        np.vstack(loc_world_hist_pd[tid_pd]['loc']),
                        cam_pose)[:, [0, 2]]
                    quat_cam_rot_t = Quaternion(matrix=cam_pose.rotation.T)
                    yaw_hist_pd = []
                    for quat_yaw_world_pd in loc_world_hist_pd[tid_pd]['yaw']:
                        rotation_cam = quat_cam_rot_t * quat_yaw_world_pd
                        vtrans = np.dot(rotation_cam.rotation_matrix,
                                        np.array([1, 0, 0]))
                        yaw_hist_pd.append(
                            -np.arctan2(vtrans[2], vtrans[0]).tolist())
                    yaw_hist_pd = np.vstack(yaw_hist_pd)
                    pu.plot_bev_obj(
                        self.ax,
                        center_pd,
                        center_hist_pd,
                        yaw_pd,
                        yaw_hist_pd,
                        l_pd,
                        w_pd,
                        color_bev,
                        'PD',
                        line_width=2)

            # Plot
            if vid_trk and (self.draw_3d or self.draw_2d):
                vid_trk.write(cv2.resize(rawimg, (self.resW, self.resH)))
            elif self.draw_3d or self.draw_2d:
                key = 0
                while (key not in [ord('q'), ord(' '), 27]):
                    cv2.imshow('preview',
                               cv2.resize(rawimg, (self.resW, self.resH)))
                    key = cv2.waitKey(1)

                if key == 27:
                    cv2.destroyAllWindows()
                    return

            # Plot
            if self.draw_bev:
                self.draw_bev_canvas()

                if vid_bev:
                    vid_bev.write(pu.fig2data(self.fig))
                    plt.cla()
                else:
                    self.fig.show()
                    plt.cla()

        if self.is_save:
            vid_trk.release()
            vid_bev.release()
        print(f"{self.res_folder} Done!")

    def save_vid(self, info_pd, info_gt):
        # Loop over save_range and plot the BEV
        print("Total {} frames. Now saving...".format(
            sum([len(seq['frames']) for _, seq in info_pd.items()])))

        # Iterate through all objects
        for ((n_seq_pd, pd_seq), (n_seq_gt,
                                  gt_seq)) in zip(info_pd.items(),
                                                  info_gt.items()):
            assert n_seq_gt == n_seq_pd

            # Plot annotation with predictions
            if self.draw_3d or self.draw_2d or self.draw_bev:
                self.plot_3D_box(n_seq_pd, pd_seq, gt_seq)

            # Merge two video vertically
            if self.is_merge:
                subfolder = 'shows_3D' if self.draw_3d else 'shows_2D'
                output_name = self.trk_vid_name.replace(
                    subfolder, 'shows_compose').replace('tracking', 'compose')
                merge_vid(self.trk_vid_name, self.bev_vid_name, output_name)

            if self.is_remove:
                os.system(f"rm {self.trk_vid_name}")
                os.system(f"rm {self.bev_vid_name}")

        print("Done!")


def main():
    args = parse_args()

    # Visualize
    visualizer = Visualizer(args.dataset, args.res_folder, args.fps,
                            args.draw_bev, args.draw_2d, args.draw_3d,
                            args.draw_traj, args.draw_tid, args.is_save,
                            args.is_merge, args.is_remove, args.is_gt,
                            args.align_gt)
    info_gt = coco_rf(args.gt_folder, cat_mapping[args.dataset])
    if args.res_folder is None:
        info_pd = info_gt
    else:
        info_pd = coco_rf(args.res_folder, cat_mapping[args.dataset])

    visualizer.save_vid(info_pd, info_gt)


if __name__ == '__main__':
    main()
