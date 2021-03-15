# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import argparse
import os
import os.path as osp
from typing import Any, Dict, List, TextIO, Tuple, Union

import motmetrics as mm
import numpy as np

from shapely.geometry.polygon import Polygon
from tqdm import tqdm
import lap

from scripts import kitti_utils as ku
from scripts.object_ap_eval.coco_format import read_file as coco_rf

mm.lap.set_default_solver(lap.lapjv)
mh = mm.metrics.create()
_PathLike = Union[str, "os.PathLike[str]"]


def in_distance_range_pose(ego_center: np.ndarray, pose: np.ndarray,
                           d_min: float, d_max: float) -> bool:
    """Determine if a pose is within distance range or not.

    Args:
        ego_center: ego center pose (zero if bbox is in ego frame).
        pose:  pose to test.
        d_min: minimum distance range
        d_max: maximum distance range

    Returns:
        A boolean saying if input pose is with specified distance range.
    """

    dist = float(np.linalg.norm(pose[0:3] - ego_center[0:3]))

    return dist > d_min and dist < d_max


def iou_polygon(poly1: Polygon, poly2: Polygon) -> float:
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return float(1 - inter / union)


def get_distance_iou_3d(x1: np.ndarray,
                        x2: np.ndarray,
                        name: str = "bbox") -> float:

    w1 = x1["width"]
    l1 = x1["length"]
    h1 = x1["height"]

    w2 = x2["width"]
    l2 = x2["length"]
    h2 = x2["height"]

    x_overlap = max(0, min(l1 / 2, l2 / 2) - max(-l1 / 2, -l2 / 2))
    y_overlap = max(0, min(w1 / 2, w2 / 2) - max(-w1 / 2, -w2 / 2))
    overlapArea = x_overlap * y_overlap
    inter = overlapArea * min(h1, h2)
    union = w1 * l1 * h1 + w2 * l2 * h2 - inter
    score = 1 - inter / union

    return float(score)


def get_orientation_error_deg(yaw1: float, yaw2: float) -> float:
    """
    Compute the smallest difference between 2 angles, in magnitude (absolute difference).
    First, find the difference between the two yaw angles; since
    each angle is guaranteed to be [-pi,pi] as the output of arctan2, then their
    difference is bounded to [-2pi,2pi].
    
    If the difference exceeds pi, then its corresponding angle in [-pi,0]
    would be smaller in magnitude. On the other hand, if the difference is
    less than -pi degrees, then we are guaranteed its counterpart in [0,pi]
    would be smaller in magnitude.
    Ref:
    https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
        Args:
        -   yaw1: angle around unit circle, in radians in [-pi,pi]
        -   yaw2: angle around unit circle, in radians in [-pi,pi]
        Returns:
        -   error: smallest difference between 2 angles, in degrees
    """
    EPSILON = 1e-5
    assert -(np.pi + EPSILON) < yaw1 and yaw1 < (np.pi + EPSILON)
    assert -(np.pi + EPSILON) < yaw2 and yaw2 < (np.pi + EPSILON)

    error = np.rad2deg(yaw1 - yaw2)
    if error > 180:
        error -= 360
    if error < -180:
        error += 360

    # get positive angle difference instead of signed angle difference
    error = np.abs(error)
    return float(error)


def get_distance(x1: np.ndarray, x2: np.ndarray, name: str) -> float:
    """Get the distance between two poses, returns nan if distance is larger than detection threshold.

    Args:
        x1: first pose
        x2: second pose
        name: name of the field to test

    Returns:
        A distance value or NaN
    """
    dist = float(np.linalg.norm(x1["centroid"][0:3] - x2["centroid"][0:3]))
    if name == "centroid":
        return dist if dist < 2 else np.nan
    elif name == "iou":
        return get_distance_iou_3d(x1, x2, name) if dist < 2 else np.nan
    elif name == "orientation":
        return get_orientation_error_deg(x1[name],
                                         x2[name]) if dist < 2 else np.nan
    else:
        raise ValueError("Not implemented..")


def create_entry(annos: Dict[str, Any], d_min: int,
                 d_max: int) -> Tuple[Dict[str, Any], List]:

    entry_dict: Dict[str, Dict[str, Any]] = {}
    id_entry = []

    for _, anno in enumerate(annos):
        location = np.array([anno['location']])
        if in_distance_range_pose(
                np.zeros(3), location.squeeze(), d_min, d_max):
            track_label_uuid = anno["track_id"]
            if track_label_uuid == -1:
                continue
            entry_dict[track_label_uuid] = {}
            # entry_dict[track_label_uuid]["bbox"] = anno['box']
            entry_dict[track_label_uuid]["centroid"] = location

            roty = ku.alpha2rot_y(anno['alpha'], location[0][0],
                                  location[0][2])
            entry_dict[track_label_uuid]["orientation"] = roty
            entry_dict[track_label_uuid]["height"] = anno['dimension'][0]
            entry_dict[track_label_uuid]["width"] = anno['dimension'][1]
            entry_dict[track_label_uuid]["length"] = anno['dimension'][2]

            id_entry.append(track_label_uuid)

    return entry_dict, id_entry


def evaluate(gt_datas: dict, pd_datas: dict, d_min: float, d_max: float,
             out_file: TextIO) -> None:
    """Evaluate tracking output.

    Args:
        gt_datas: path to dataset
        pd_datas: list of path to tracker output
        d_min: minimum distance range
        d_max: maximum distance range
        out_file: output file object
    """
    acc_c = mm.MOTAccumulator(auto_id=True)
    acc_i = mm.MOTAccumulator(auto_id=True)
    acc_o = mm.MOTAccumulator(auto_id=True)

    ID_gt_all: List[str] = []

    count_all: int = 0
    fr_count: int = 0

    tqdm.write(f"{len(pd_datas)} {len(gt_datas)}")
    assert len(pd_datas) == len(gt_datas)

    pbar = tqdm(zip(pd_datas.items(), gt_datas.items()), total=len(gt_datas))
    for (log_id_pd, pd_data), (log_id_gt, gt_data) in pbar:
        fr_count += len(pd_data['frames'])
        pbar.set_postfix_str(s=f"Logs: {log_id_gt} AccumFrames: {fr_count} | "
                             f"PD: {len(pd_data['frames'])} "
                             f"GT: {len(gt_data['frames'])}]")

        assert len(pd_data['frames']) == len(gt_data['frames'])
        assert log_id_pd == log_id_gt

        for (_, hypos), (_, annos) in \
                zip(pd_data['frames'].items(), gt_data['frames'].items()):

            # Get entries in GT and PD
            gt, id_gts = create_entry(annos['annotations'], d_min, d_max)
            tracks, id_tracks = create_entry(hypos['annotations'], d_min,
                                             d_max)

            ID_gt_all.append(np.unique(id_gts).tolist())

            dists_c: List[List[float]] = []
            dists_i: List[List[float]] = []
            dists_o: List[List[float]] = []
            for _, gt_value in gt.items():
                gt_track_data_c: List[float] = []
                gt_track_data_i: List[float] = []
                gt_track_data_o: List[float] = []
                dists_c.append(gt_track_data_c)
                dists_i.append(gt_track_data_i)
                dists_o.append(gt_track_data_o)
                for _, track_value in tracks.items():
                    count_all += 1
                    gt_track_data_c.append(
                        get_distance(gt_value, track_value, "centroid"))
                    gt_track_data_i.append(
                        get_distance(gt_value, track_value, "iou"))
                    gt_track_data_o.append(
                        get_distance(gt_value, track_value, "orientation"))

            acc_c.update(id_gts, id_tracks, dists_c)
            acc_i.update(id_gts, id_tracks, dists_i)
            acc_o.update(id_gts, id_tracks, dists_o)

    ID_gt_all = np.unique([item for lists in ID_gt_all for item in lists])

    if count_all == 0:
        # fix for when all hypothesis is empty,
        # pymotmetric currently doesn't support this, see https://github.com/cheind/py-motmetrics/issues/49
        acc_c.update(id_gts, [-1], np.ones(np.shape(id_gts)) * np.inf)
        acc_i.update(id_gts, [-1], np.ones(np.shape(id_gts)) * np.inf)
        acc_o.update(id_gts, [-1], np.ones(np.shape(id_gts)) * np.inf)

    tqdm.write("Computing...")
    summary = mh.compute(
        acc_c,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "mostly_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
        ],
        name="acc",
    )
    tqdm.write(f"summary = \n{summary}")
    num_tracks = len(ID_gt_all)
    if num_tracks == 0:
        num_tracks = 1

    num_frames = summary["num_frames"][0]
    mota = summary["mota"][0] * 100
    motp_c = summary["motp"][0]
    idf1 = summary["idf1"][0]
    most_track = summary["mostly_tracked"][0] / num_tracks
    most_lost = summary["mostly_lost"][0] / num_tracks
    num_fp = summary["num_false_positives"][0]
    num_miss = summary["num_misses"][0]
    num_switch = summary["num_switches"][0]
    num_frag = summary["num_fragmentations"][0]

    #acc_c.events.loc[acc_c.events.Type != "RAW",
    #                 "D"] = acc_i.events.loc[acc_c.events.Type != "RAW", "D"]

    sum_motp_i = mh.compute(acc_i, metrics=["motp"], name="acc")
    tqdm.write(f"MOTP-I = \n{sum_motp_i}")

    motp_i = sum_motp_i["motp"][0]

    # acc_c.events.loc[acc_c.events.Type != "RAW",
    #                 "D"] = acc_o.events.loc[acc_c.events.Type != "RAW", "D"]
    sum_motp_o = mh.compute(acc_o, metrics=["motp"], name="acc")
    tqdm.write(f"MOTP-O = \n{sum_motp_o}")

    motp_o = sum_motp_o["motp"][0]

    out_string = (f"{num_frames} {mota:.2f} "
                  f"{motp_c:.2f} {motp_o:.2f} {motp_i:.2f} "
                  f"{idf1:.2f} {most_track:.2f} {most_lost:.2f} "
                  f"{num_fp} {num_miss} {num_switch} {num_frag}\n")
    out_file.write(out_string)


def eval_tracks(path_seq_map: _PathLike, path_gt: _PathLike,
                path_pd: _PathLike, category: list, d_min: float, d_max: float,
                out_filename: str) -> None:

    if not os.path.exists(path_gt):
        tqdm.write(f"Missing {path_gt}")

    gt_datas = coco_rf(path_gt, category)
    path_pd = osp.join(args.res_folder, '..', 'output.json')
    pd_datas = coco_rf(path_pd, category)

    with open(out_filename, "w") as out_file:
        evaluate(gt_datas, pd_datas, d_min, d_max, out_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gt_map', help='groundtruth seqmap')
    parser.add_argument('gt_folder', help='groundtruth label folder')
    parser.add_argument('res_folder', help='results folder')
    parser.add_argument(
        "--category",
        nargs="+",
        default=["Car"],
        choices=["Car", "Pedestrian"])
    parser.add_argument("--flag", type=str)
    parser.add_argument("--d_min", type=float, default=0)
    parser.add_argument("--d_max", type=float, default=100, required=True)

    args = parser.parse_args()
    tqdm.write(f"args = {args}")

    out_filename = os.path.join(
        args.res_folder,
        f"{args.flag}_{int(args.d_min)}_{int(args.d_max)}.txt")
    tqdm.write(f"output file name = {out_filename}")

    eval_tracks(
        args.gt_map,
        args.gt_folder,
        args.res_folder,
        args.category,
        args.d_min,
        args.d_max,
        out_filename,
    )
