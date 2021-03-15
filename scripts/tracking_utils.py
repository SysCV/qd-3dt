import numba
import numpy as np
from numpy.linalg import inv

import math
import cv2
import sklearn.metrics.pairwise as skp
from scipy.spatial.transform import Rotation as R
import torch
from shapely.geometry import Polygon


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, verbose=False):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            if verbose:
                print("Avg: {} | Sum: {} | Count: {}".format(
                    self.avg, self.sum, self.count))
            self.avg = self.sum / self.count
        else:
            print("Not update! count equals 0")


def convert_3dbox_to_8corner(bbox3d: torch.tensor) -> torch.tensor:
    '''Get 3D bbox vertex in camera coordinates
        Args:
        -   bbox3d: torch tensor of shape (7,) representing
                tx,ty,tz,yaw,l,w,h. (tx,ty,tz,yaw) tell us how to
                transform points to get from the object frame to 
                the egovehicle frame.
        Returns:
        -   corners_3d: (8,3) array in egovehicle frame
    '''
    # box location in camera coordinates
    loc = bbox3d[:3]

    # 3d bounding box dimensions
    dim = bbox3d[4:]

    # compute rotational matrix around yaw axis
    yaw = bbox3d[3]
    R = bbox3d.new_tensor([[+torch.cos(yaw), 0, +torch.sin(yaw)], [0, 1, 0],
                           [-torch.sin(yaw), 0, +torch.cos(yaw)]])
    corners = get_vertex_torch(dim)

    # rotate and translate 3d bounding box
    corners = corners.mm(R.t()) + loc
    return corners


def get_vertex_torch(box_dim: torch.tensor):
    '''Get 3D bbox vertex (used for the upper volume iou calculation)
    Input:
        box_dim: a tuple of (h, w, l)
    Output:
        vertex: torch tensor of shape (8, 3) for bbox vertex
    '''
    h, w, l = box_dim
    # 3d bounding box corners
    corners = box_dim.new_tensor(
        [[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
         [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
         [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]])
    return corners.t()


def yaw2alpha_torch(rot_y, x_loc, z_loc):
    """
    Get alpha by rotation_y - theta
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    torch_pi = rot_y.new_tensor([np.pi])
    alpha = rot_y - torch.atan2(x_loc, z_loc)
    alpha = (alpha + torch_pi) % (2 * torch_pi) - torch_pi
    return alpha


def alpha2yaw_torch(alpha, x_loc, z_loc):
    """
    Get rotation_y by alpha + theta
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    torch_pi = alpha.new_tensor([np.pi])
    rot_y = alpha + torch.atan2(x_loc, z_loc)
    rot_y = (rot_y + torch_pi) % (2 * torch_pi) - torch_pi
    return rot_y


def point3dcoord_torch(points, depths, projection, position, rotation):
    """
    project point to 3D world coordinate

    point: (N, 2), N points on X-Y image plane
    projection: (3, 4), projection matrix
    position:  (3), translation of world coordinates
    rotation:  (3, 3), rotation along camera coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    assert points.shape[1] == 2, ("Shape ({}) not fit".format(points.shape))
    corners = imagetocamera_torch(points, depths, projection)
    corners_global = cameratoworld_torch(corners, position, rotation)
    return corners_global


def imagetocamera_torch(points, depths, projection):
    """
    points: (N, 2), N points on X-Y image plane
    depths: (N,), N depth values for points
    projection: (3, 4), projection matrix

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert points.shape[1] == 2, "Shape ({}) not fit".format(points.shape)
    corners = torch.cat([points, points.new_ones((points.shape[0], 1))],
                        dim=1).mm(projection[:, 0:3].inverse().t())
    assert torch.all(abs(corners[:, 2] - 1) < 0.01)
    corners_cam = corners * depths.view(-1, 1)

    return corners_cam


def cameratoimage_torch(corners, projection, invalid_value=-1000):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera plane
    projection: (3, 4), projection matrix

    points: (N, 2), N points on X-Y image plane
    """
    assert corners.shape[1] == 3, "Shape ({}) not fit".format(corners.shape)

    points = torch.cat(
        [corners, corners.new_ones(
            (corners.shape[0], 1))], dim=1).mm(projection.t())

    # [x, y, z] -> [x/z, y/z]
    mask = points[:, 2:3] > 0
    points_img = (points[:, :2] / points[:, 2:3]
                  ) * mask + invalid_value * torch.logical_not(mask)

    return points_img


def cameratoworld_torch(corners, position, rotation):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    assert corners.shape[1] == 3, ("Shape ({}) not fit".format(corners.shape))
    corners_global = corners.mm(rotation.t()) + position[None]
    return corners_global


def worldtocamera_torch(corners_global, position, rotation):
    """
    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert corners_global.shape[1] == 3, ("Shape ({}) not fit".format(
        corners_global.shape))
    corners = (corners_global - position[None]).mm(rotation)
    return corners


def point3dcoord(points, depths, projection, pose):
    """
    project point to 3D world coordinate

    point: (N, 2), N points on X-Y image plane
    depths: (N, 1), N depth values
    projection: (3, 4), projection matrix
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    assert points.shape[1] == 2, ("Shape ({}) not fit".format(points.shape))
    corners = imagetocamera(points, depths, projection)
    corners_global = cameratoworld(corners, pose)
    return corners_global


def boxto3dcoord(box, depth, projection, pose):
    """
    project a box center to 3D world coordinate

    box: (5,), N boxes on X-Y image plane
    projection: (3, 4), projection matrix
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (3, 1), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    x1, y1, x2, y2 = box[:4]
    points = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2]])
    return point3dcoord(points, depth, projection, pose)


def projection3d(projection, pose, corners_global):
    """
    project 3D point in world coordinate to 2D image plane

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    projection: (3, 4), projection matrix
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    point: (N, 2), N points on X-Y image plane
    """
    corners = worldtocamera(corners_global, pose)
    corners = cameratoimage(corners, projection)
    return corners


def cameratoimage(corners, projection, invalid_value=-1000):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera plane
    projection: (3, 4), projection matrix

    points: (N, 2), N points on X-Y image plane
    """
    assert corners.shape[1] == 3, "Shape ({}) not fit".format(corners.shape)

    points = np.hstack([corners, np.ones(
        (corners.shape[0], 1))]).dot(projection.T)

    # [x, y, z] -> [x/z, y/z]
    mask = points[:, 2:3] > 0
    points = (points[:, :2] / points[:, 2:3]) * mask + invalid_value * (1 -
                                                                        mask)

    return points


def imagetocamera(points, depth, projection):
    """
    points: (N, 2), N points on X-Y image plane
    depths: (N,), N depth values for points
    projection: (3, 4), projection matrix

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert points.shape[1] == 2, "Shape ({}) not fit".format(points.shape)

    corners = np.hstack([points, np.ones(
        (points.shape[0], 1))]).dot(inv(projection[:, 0:3]).T)
    assert np.allclose(corners[:, 2], 1)
    corners *= depth.reshape(-1, 1)

    return corners


def worldtocamera(corners_global, pose):
    """
    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert corners_global.shape[1] == 3, ("Shape ({}) not fit".format(
        corners_global.shape))
    corners = (corners_global - pose.position[np.newaxis]).dot(pose.rotation)
    return corners


def cameratoworld(corners, pose):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    assert corners.shape[1] == 3, ("Shape ({}) not fit".format(corners.shape))
    corners_global = corners.dot(pose.rotation.T) + \
                     pose.position[np.newaxis]
    return corners_global


def compute_boxoverlap_with_depth(det,
                                  detbox,
                                  detdepth,
                                  detdim,
                                  trkdet,
                                  trkedboxes,
                                  trkeddepths,
                                  trkdims,
                                  H=1080,
                                  W=1920):
    iou_2d = np.zeros((len(trkeddepths)))

    # Sum up all the available region of each tracker
    for i in range(len(trkedboxes)):
        iou_2d[i] = compute_iou(trkedboxes[i], detbox)
    depth_weight = np.exp(-abs(trkeddepths - detdepth) / (detdepth + 1e-6))

    #print(iou_2d, depth_weight, iou_2d*depth_weight)
    # Calculate the IOU
    iou_2d *= depth_weight
    # Check if depth difference is within the diagonal distance of two cars
    iou_2d *= (detdim[2] + detdim[1] + trkdims[:, 2] + trkdims[:, 1]) > \
                abs(detdepth - trkeddepths)
    return iou_2d


def compute_boxoverlap_with_depth_draw(det,
                                       detbox,
                                       detdepth,
                                       detdim,
                                       trkdet,
                                       trkedboxes,
                                       trkeddepths,
                                       trkdims,
                                       H=1080,
                                       W=1920):
    overlap = np.zeros((len(trkeddepths)))
    valid_trkbox = np.zeros((len(trkeddepths)))
    iou_2d = np.zeros((len(trkeddepths)))
    same_layer = np.zeros((len(trkeddepths)))
    # Find where the DOI is in the tracking depth order
    idx = np.searchsorted(trkeddepths, detdepth)

    boxesbefore = trkedboxes[:idx]
    boxesafter = trkedboxes[idx:]
    before = np.zeros((H, W), dtype=np.uint8)
    after = np.zeros((H, W), dtype=np.uint8)
    now = np.zeros((H, W), dtype=np.uint8)
    # Plot 2D bounding box according to the depth order
    for idx, box in enumerate(boxesbefore):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)
        before = cv2.rectangle(before, (x1, y1), (x2, y2), idx + 1, -1)
    for idx, box in enumerate(reversed(boxesafter)):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)
        after = cv2.rectangle(after, (x1, y1), (x2, y2),
                              len(trkedboxes) - idx, -1)

    x1, y1, x2, y2, _ = detbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    now = cv2.rectangle(now, (x1, y1), (x2, y2), 1, -1)

    currentpixels = np.where(now == 1)
    pixelsbefore = before[currentpixels]
    pixelsafter = after[currentpixels]

    # Sum up all the available region of each tracker
    for i in range(len(trkedboxes)):
        overlap[i] = np.sum(pixelsbefore == (i + 1)) + np.sum(
            pixelsafter == (i + 1))
        iou_2d[i] = compute_iou(trkdet[i], det)
        same_layer[i] = np.sum((abs(trkeddepths[i] - trkeddepths) < 1)) > 1
        valid_trkbox[i] = np.sum(before == (i + 1)) + np.sum(after == (i + 1))
    #trkedboxesarr = np.array(trkedboxes).astype('float')
    overlap = overlap.astype('float')

    # Calculate the IOU
    #trkareas = (trkedboxesarr[:, 2] - trkedboxesarr[:, 0]) * (
    #        trkedboxesarr[:, 3] - trkedboxesarr[:, 1])
    trkareas = valid_trkbox
    trkareas += (x2 - x1) * (y2 - y1)
    trkareas -= overlap
    occ_iou = overlap / (trkareas + (trkareas == 0).astype(int))
    occ_iou[occ_iou > 1.0] = 1.0

    #print(occ_iou, iou_2d, same_layer)
    occ_iou += same_layer * (iou_2d - occ_iou)
    # Check if depth difference is within the diagonal distance of two cars
    occ_iou *= (detdim[2] + detdim[1] + trkdims[:, 2] + trkdims[:, 1]) > \
                abs(detdepth - trkeddepths)
    return occ_iou


def construct2dlayout(trks, dims, rots, cam_calib, pose, cam_near_clip=0.15):
    depths = []
    boxs = []
    points = []
    corners_camera = worldtocamera(trks, pose)
    for corners, dim, rot in zip(corners_camera, dims, rots):
        # in camera coordinates
        points3d = computeboxes(rot, dim, corners)
        depths.append(corners[2])
        projpoints = get_3d_bbox_vertex(cam_calib, pose, points3d, cam_near_clip)
        points.append(projpoints)
        if projpoints == []:
            box = np.array([-1000, -1000, -1000, -1000])
            boxs.append(box)
            depths[-1] = -10
            continue
        projpoints = np.vstack(projpoints)[:, :2]
        projpoints = projpoints.reshape(-1, 2)
        minx = projpoints[:, 0].min()
        maxx = projpoints[:, 0].max()
        miny = projpoints[:, 1].min()
        maxy = projpoints[:, 1].max()
        box = np.array([minx, miny, maxx, maxy])
        boxs.append(box)
    return boxs, depths, points


def computeboxes(roty, dim, loc):
    '''Get 3D bbox vertex in camera coordinates 
    Input:
        roty: (1,), object orientation, -pi ~ pi
        box_dim: a tuple of (h, w, l)
        loc: (3,), box 3D center
    Output:
        vertex: numpy array of shape (8, 3) for bbox vertex
    '''
    roty = roty[0]
    R = np.array([[+np.cos(roty), 0, +np.sin(roty)], [0, 1, 0],
                  [-np.sin(roty), 0, +np.cos(roty)]])
    corners = get_vertex(dim)
    corners = corners.dot(R.T) + loc
    return corners


def get_vertex(box_dim):
    '''Get 3D bbox vertex (used for the upper volume iou calculation)
    Input:
        box_dim: a tuple of (h, w, l)
    Output:
        vertex: numpy array of shape (8, 3) for bbox vertex
    '''
    h, w, l = box_dim
    corners = np.array(
        [[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
         [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
         [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]])
    return corners.T


def get_3d_bbox_vertex(cam_calib, cam_pose, points3d, cam_near_clip=0.15):
    '''Get 3D bbox vertex in camera coordinates 
    Input:
        cam_calib: (3, 4), projection matrix
        cam_pose: a class with position, rotation of the frame
            rotation:  (3, 3), rotation along camera coordinates
            position:  (3), translation of world coordinates
        points3d: (8, 3), box 3D center in camera coordinates
        cam_near_clip: in meter, distance to the near plane
    Output:
        points: numpy array of shape (8, 2) for bbox in image coordinates
    '''
    lineorder = np.array(
        [
            [1, 2, 6, 5],  # front face
            [2, 3, 7, 6],  # left face
            [3, 4, 8, 7],  # back face
            [4, 1, 5, 8],
            [1, 6, 5, 2]
        ],
        dtype=np.int32) - 1  # right

    points = []

    # In camera coordinates
    cam_dir = np.array([0, 0, 1])
    center_pt = cam_dir * cam_near_clip

    for i in range(len(lineorder)):
        for j in range(4):
            p1 = points3d[lineorder[i, j]].copy()
            p2 = points3d[lineorder[i, (j + 1) % 4]].copy()

            before1 = is_before_clip_plane_camera(p1[np.newaxis],
                                                  cam_near_clip)[0]
            before2 = is_before_clip_plane_camera(p2[np.newaxis],
                                                  cam_near_clip)[0]

            inter = get_intersect_point(center_pt, cam_dir, p1, p2)

            if not (before1 or before2):
                # print("Not before 1 or 2")
                continue
            elif before1 and before2:
                # print("Both 1 and 2")
                vp1 = p1
                vp2 = p2
            elif before1 and not before2:
                # print("before 1 not 2")
                vp1 = p1
                vp2 = inter
            elif before2 and not before1:
                # print("before 2 not 1")
                vp1 = inter
                vp2 = p2

            cp1 = cameratoimage(vp1[np.newaxis], cam_calib)[0]
            cp2 = cameratoimage(vp2[np.newaxis], cam_calib)[0]
            points.append((cp1, cp2))
    return points


def compute_cos_dis(featA, featB):
    return np.exp(-skp.pairwise_distances(featA, featB))


def compute_cos_sim(featA, featB):
    sim = np.dot(featA, featB.T)
    sim /= np.linalg.norm(featA, axis=1).reshape(featA.shape[0], 1)
    sim /= np.linalg.norm(featB, axis=1).reshape(featB.shape[0], 1).T
    return sim


@numba.jit(nopython=True, nogil=True)
def compute_iou(boxA, boxB):
    if boxA[0] > boxB[2] or boxB[0] > boxA[2] or boxA[1] > boxB[3] \
            or boxB[1] > boxA[3]:
        return 0
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def compute_iou_arr(boxA, boxB):
    """
    compute IOU in batch format without for-loop
    NOTE: only work in normalized coordinate (x, y in [0, 1])
    boxA: a array of box with shape [N1, 4]
    boxB: a array of box with shape [N2, 4]

    return a array of IOU with shape [N1, N2]
    """
    boxBt = boxB.transpose()

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:, 0:1], boxBt[0:1, :])
    yA = np.maximum(boxA[:, 1:2], boxBt[1:2, :])
    xB = np.minimum(boxA[:, 2:3], boxBt[2:3, :])
    yB = np.minimum(boxA[:, 3:4], boxBt[3:4, :])

    # compute the area of intersection rectangle
    x_diff = np.maximum(xB - xA, 0)
    y_diff = np.maximum(yB - yA, 0)
    interArea = (x_diff + 1) * (y_diff + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:, 2:3] - boxA[:, 0:1] + 1) * \
               (boxA[:, 3:4] - boxA[:, 1:2] + 1)
    boxBArea = (boxBt[2:3, :] - boxBt[0:1, :] + 1) * \
               (boxBt[3:4, :] - boxBt[1:2, :] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_iou(box_t1, box_t2, thres):
    """
    Input:
        box_t1: (N1, 4)
        box_t2: (N2, 4)
        thres: single value
    Output:
        iou: Float (N1, N2)
        idx: Long (N1, 1)
        valid: Float (N1, 1)
    """
    # Get IOU
    iou_tensor = compute_iou_arr(box_t1, box_t2)  # [x1, y1, x2, y2]

    # Select index
    val = np.max(iou_tensor, axis=1)
    idx = iou_tensor.argmax(axis=1)

    # Matched index
    valid = (val > thres).reshape(-1, 1)

    return iou_tensor, idx, valid


def matching(box_t1, box_t2, thres=0.85):
    """
    Match input order by box IOU, select matched feature and box at time t2.
    The match policy is as follows:
        time t1: matched - USE
                 not matched - DUPLICATE ITSELF
        time t2: matched - USE
                 not matched - DISCARD
    Use idx as a match order index to swap their order
    Use valid as a valid threshold to keep size by duplicating t1 itself

    Input:
        box_t1: (N1, 4)
        box_t2: (N2, 4)

    Inter:
        iou: Float
        matches: Float
        idx: Long
        valid: Float

    Output:
        new_box_t2: (N1, 4)
    """

    if box_t1.size == 0 or box_t2.size == 0:
        return None, None, None

    # Get IOU
    iou, idx, valid = get_iou(box_t1, box_t2, thres)

    # Select features
    new_box_t2 = box_t2[idx] * valid + box_t1 * (1 - valid)

    return new_box_t2, idx, valid


@numba.jit()
def rad2deg(rad):
    return rad * 180.0 / np.pi


@numba.jit()
def deg2rad(deg):
    return deg / 180.0 * np.pi


@numba.jit()
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


@numba.jit()
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


@numba.jit(nopython=True, nogil=True)
def rot_axis(angle, axis):
    # RX = np.array([ [1,             0,              0],
    #                 [0, np.cos(gamma), -np.sin(gamma)],
    #                 [0, np.sin(gamma),  np.cos(gamma)]])
    #
    # RY = np.array([ [ np.cos(beta), 0, np.sin(beta)],
    #                 [            0, 1,            0],
    #                 [-np.sin(beta), 0, np.cos(beta)]])
    #
    # RZ = np.array([ [np.cos(alpha), -np.sin(alpha), 0],
    #                 [np.sin(alpha),  np.cos(alpha), 0],
    #                 [            0,              0, 1]])
    cg = np.cos(angle)
    sg = np.sin(angle)
    if axis == 0:  # X
        v = [0, 4, 5, 7, 8]
    elif axis == 1:  # Y
        v = [4, 0, 6, 2, 8]
    else:  # Z
        v = [8, 0, 1, 3, 4]
    RX = np.zeros(9, dtype=numba.float64)
    RX[v[0]] = 1.0
    RX[v[1]] = cg
    RX[v[2]] = -sg
    RX[v[3]] = sg
    RX[v[4]] = cg
    return RX.reshape(3, 3)


# Same as angle2rot from kio_slim
def angle2rot(rotation, inverse=False):
    return rotate(np.eye(3), rotation, inverse=inverse)


@numba.jit(nopython=True, nogil=True)
def rotate(vector, angle, inverse=False):
    """
    Rotation of x, y, z axis
    Forward rotate order: Z, Y, X
    Inverse rotate order: X^T, Y^T,Z^T
    Input:
        vector: vector in 3D coordinates
        angle: rotation along X, Y, Z (raw data from GTA)
    Output:
        out: rotated vector
    """
    gamma, beta, alpha = angle[0], angle[1], angle[2]

    # Rotation matrices around the X (gamma), Y (beta), and Z (alpha) axis
    RX = rot_axis(gamma, 0)
    RY = rot_axis(beta, 1)
    RZ = rot_axis(alpha, 2)

    # Composed rotation matrix with (RX, RY, RZ)
    if inverse:
        return np.dot(np.dot(np.dot(RX.T, RY.T), RZ.T), vector)
    else:
        return np.dot(np.dot(np.dot(RZ, RY), RX), vector)


def rotate_(vector, angle, inverse=False):
    """Rotation of x, y, z axis
        Forward rotate order: Z, Y, X
        Inverse rotate order: X^T, Y^T,Z^T

    Args:
        vector (np.ndarray): vector in 3D coordinates
        angle (np.ndarray): rotation along X, Y, Z (raw data from GTA)
        inverse (bool, optional): Inverse the rotation matrix. Defaults to False.

    Returns:
        np.ndarray: rotated vector
    """
    rotmat = R.from_euler('xyz', angle).as_matrix()
    if inverse:
        return rotmat.T.dot(vector)
    else:
        return rotmat.dot(vector)


def cal_3D_iou(vol_box_pd, vol_box_gt):
    vol_inter = intersect_bbox_with_yaw(vol_box_pd, vol_box_gt)
    vol_gt = intersect_bbox_with_yaw(vol_box_gt, vol_box_gt)
    vol_pd = intersect_bbox_with_yaw(vol_box_pd, vol_box_pd)

    return get_vol_iou(vol_pd, vol_gt, vol_inter)


def intersect_bbox_with_yaw(box_a, box_b):
    """
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
    """
    # height (Z) overlap
    min_h_a = np.min(box_a[2])
    max_h_a = np.max(box_a[2])
    min_h_b = np.min(box_b[2])
    max_h_b = np.max(box_b[2])
    max_of_min = np.max([min_h_a, min_h_b])
    min_of_max = np.min([max_h_a, max_h_b])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        # print("Z = 0")
        return 0.

    # oriented XY overlap
    # TODO: Check if the order of 3D box is correct
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        # print("XY = 0")
        return 0.

    return z_intersection * xy_intersection


@numba.jit(nopython=True, nogil=True)
def get_vol_iou(vol_a, vol_b, vol_intersect):
    union = vol_a + vol_b - vol_intersect
    return vol_intersect / union if union else 0.


@numba.jit(nopython=True)
def get_intersect_point(center_pt, cam_dir, vertex1, vertex2):
    # get the intersection point of two 3D points and a plane
    c1 = center_pt[0]
    c2 = center_pt[1]
    c3 = center_pt[2]
    a1 = cam_dir[0]
    a2 = cam_dir[1]
    a3 = cam_dir[2]
    x1 = vertex1[0]
    y1 = vertex1[1]
    z1 = vertex1[2]
    x2 = vertex2[0]
    y2 = vertex2[1]
    z2 = vertex2[2]

    k_up = abs(a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3))
    k_down = abs(a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2))
    if k_up > k_down:
        k = 1
    else:
        k = k_up / k_down
    inter_point = (1 - k) * vertex1 + k * vertex2
    return inter_point


def is_before_clip_plane_world(points_world, cam_pose, cam_near_clip=0.15):
    """
    points_world: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates
    cam_near_clip: scalar, the near projection plane

    is_before: (N,), bool, is the point locate before the near clip plane
    """
    return worldtocamera(points_world, cam_pose)[:, 2] > cam_near_clip


def is_before_clip_plane_camera(points_camera, cam_near_clip=0.15):
    """
    points_camera: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    cam_near_clip: scalar, the near projection plane

    is_before: bool, is the point locate before the near clip plane
    """
    return points_camera[:, 2] > cam_near_clip


def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]


def quaternion_to_euler(w, x, y, z):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


if __name__ == '__main__':
    cam_rotation = np.array([45.0, 0.0, 0.0])
    theta = (np.pi / 180.0) * cam_rotation
    vector = np.array([1., 0., 0.])

    print(rotate(np.eye(3), theta, inverse=False))
    print(angle2rot(theta, inverse=False))
    print(R.from_euler('zyx', theta).as_matrix())
    print(R.from_euler('xyz', theta).as_matrix())
    print(
        np.allclose(
            rotate(rotate(vector, theta, inverse=True), theta), vector))
