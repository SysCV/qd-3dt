import os
import numpy as np

import utm

import scripts.tracking_utils as tu


class Pose:
    ''' Calibration matrices in KITTI
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
        X (z) ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord (KITTI): 
        front x, left y, up z

        world coord (GTA):
        right x, front y, up z
        
        velodyne coord (nuScenes):
        right x, front y, up z
        
        velodyne coord (Waymo):
        front x, left y, up z

        rect/ref camera coord (KITTI, GTA, nuScenes):
        right x, down y, front z

        camera coord (Waymo):
        front x, left y, up z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    '''

    def __init__(self, position, rotation):
        # relative position to the 1st frame: (X, Y, Z)
        # relative rotation to the previous frame: (r_x, r_y, r_z)
        self.position = position
        if rotation.shape == (3, 3):
            # rotation matrices already
            self.rotation = rotation
        else:
            # rotation vector
            self.rotation = tu.angle2rot(np.array(rotation))


# Functions from kio_slim
class KittiPoseParser:

    def __init__(self, fields=None):
        self.latlon = None
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.position = None
        self.rotation = None
        if fields is not None:
            self.set_oxt(fields)

    def set_oxt(self, fields):
        fields = [float(f) for f in fields]
        self.latlon = fields[:2]
        location = utm.from_latlon(*self.latlon)
        self.position = np.array([location[0], location[1], fields[2]])

        self.roll = fields[3]
        self.pitch = fields[4]
        self.yaw = fields[5]
        rotation = tu.angle2rot(np.array([self.roll, self.pitch, self.yaw]))
        imu_to_camera = tu.angle2rot(
            np.array([np.pi / 2, -np.pi / 2, 0]), inverse=True)
        self.rotation = rotation.dot(imu_to_camera)


def rad2deg(rad):
    return rad * 180.0 / np.pi


def deg2rad(deg):
    return deg / 180.0 * np.pi


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


def alpha2rot_y(alpha, x, FOCAL_LENGTH):
    """
    Get rotation_y by alpha + theta
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x, FOCAL_LENGTH)
    rot_y = (rot_y + np.pi) % (2 * np.pi) - np.pi
    return rot_y


def parse_seq_map(path_seq_map: str):
    """get #sequences and #frames per sequence from test mapping

    Args:
        path_seq_map (str): path to the seq_map file

    Returns:
        sequence_name (list): name of each sequence
        n_frames (list): number of frames in each sequence
    """
    n_frames = []
    sequence_name = []
    with open(path_seq_map, "r") as fh:
        for lines in fh:
            fields = lines.split(" ")
            sequence_name.append(fields[0])
            n_frames.append(int(fields[3]) - int(fields[2]) + 1)
    return sequence_name, n_frames


def read_oxts(oxts_dir, seq_idx):
    """ Read oxts file and return each fields for KittiPoseParser
        e.g., 
            fields = read_oxts(oxt_dir, vid_id)
            poses = [KittiPoseParser(field) for field in fields]
    """
    oxts_path = os.path.join(oxts_dir, f'{seq_idx:04d}.txt')
    with open(oxts_path, 'r') as f:
        fields = [line.strip().split() for line in f]
    return fields


def read_calib(calib_dir, seq_idx, cam=2):
    """ Read calibration file and return camera matrix
        e.g.,
            projection = read_calib(cali_dir, vid_id)
    """
    with open(os.path.join(calib_dir, f'{seq_idx:04d}.txt')) as f:
        fields = [line.split() for line in f]
    return np.asarray(fields[cam][1:], dtype=np.float32).reshape(3, 4)


def read_calib_det(calib_dir, img_idx, cam=2):
    """ Read calibration file and return camera matrix
        e.g.,
            projection = read_calib(cali_dir, img_id)
    """
    with open(os.path.join(calib_dir, f'{img_idx:06d}.txt')) as f:
        fields = [line.split() for line in f]
    return np.asarray(fields[cam][1:], dtype=np.float32).reshape(3, 4)
