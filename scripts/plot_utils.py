import matplotlib
matplotlib.use(matplotlib.get_backend())

import cv2
import os
import os.path as osp
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sns.set(style="darkgrid")


class RandomColor():

    def __init__(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a 
        distinct 
        RGB color; the keyword argument name must be a standard mpl colormap 
        name.'''
        self.cmap = plt.cm.get_cmap(name, n)
        self.n = n

    def get_random_color(self, scale=1):
        ''' Using scale = 255 for opencv while scale = 1 for matplotlib '''
        return tuple(
            [scale * x for x in self.cmap(np.random.randint(self.n))[:3]])


def fig2data(fig, size: tuple = None):
    fig.canvas.figure.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()

    # canvas.tostring_argb give pixmap in ARGB mode.
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)

    buf.shape = (h, w, 4)  # last dim: (alpha, r, g, b)

    # Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 3, axis=2)

    # Take only RGB
    buf = buf[:, :, 1:]

    if size is not None:
        buf = cv2.resize(buf, size)

    # Get BGR from RGB
    buf = buf[:, :, ::-1]

    return buf


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(
            name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_depth(epoch, session, targets, inputs, outputs):
    fig = plt.figure(dpi=100)
    params = {
        'legend.fontsize': 'x-large',
        'figure.figsize': (15, 5),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    plt.rcParams.update(params)
    plt.title('Depth estimation', fontsize=30)

    # Plot only valid locations
    # valid = (targets != 0)
    # targets = targets[valid]
    # inputs = inputs[valid]
    # outputs = outputs[valid]
    t = np.arange(targets.shape[0])

    plt.plot(t, targets, color='g', marker='o', linewidth=2.0, label='GT')
    plt.plot(t, inputs, color='b', marker='o', linewidth=1.0, label='INPUT')
    plt.plot(t, outputs, color='r', marker='o', linewidth=1.0, label='OUTPUT')

    plt.legend()
    plt.savefig(
        'output/lstm/{}_{}_depth.eps'.format(session, epoch), format='eps')
    plt.close()


def plot_bev_obj(ax: plt.axes,
                 center: np.ndarray,
                 center_hist: np.ndarray,
                 yaw: np.ndarray,
                 yaw_hist: np.ndarray,
                 l: float,
                 w: float,
                 color: list,
                 text: str,
                 line_width: int = 1):
    # Calculate length, width of object
    vec_l = [l * np.cos(yaw), -l * np.sin(yaw)]
    vec_w = [-w * np.cos(yaw - np.pi / 2), w * np.sin(yaw - np.pi / 2)]
    vec_l = np.array(vec_l)
    vec_w = np.array(vec_w)

    # Make 4 points
    p1 = center + 0.5 * vec_l - 0.5 * vec_w
    p2 = center + 0.5 * vec_l + 0.5 * vec_w
    p3 = center - 0.5 * vec_l + 0.5 * vec_w
    p4 = center - 0.5 * vec_l - 0.5 * vec_w

    # Plot object
    line_style = '-' if 'PD' in text else ':'
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            line_style,
            c=color,
            linewidth=3 * line_width)
    ax.plot([p1[0], p4[0]], [p1[1], p4[1]],
            line_style,
            c=color,
            linewidth=line_width)
    ax.plot([p3[0], p2[0]], [p3[1], p2[1]],
            line_style,
            c=color,
            linewidth=line_width)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]],
            line_style,
            c=color,
            linewidth=line_width)

    # Plot center history
    for index, ct in enumerate(center_hist):
        yaw = yaw_hist[index].item()
        vec_l = np.array([l * np.cos(yaw), -l * np.sin(yaw)])
        ct_dir = ct + 0.5 * vec_l
        alpha = max(float(index) / len(center_hist), 0.5)
        ax.plot([ct[0], ct_dir[0]], [ct[1], ct_dir[1]],
                line_style,
                alpha=alpha,
                c=color,
                linewidth=line_width)
        ax.scatter(
            ct[0],
            ct[1],
            alpha=alpha,
            c=np.array([color]),
            linewidth=line_width)


def plot_3D(save_dir,
            epoch,
            session,
            cam_loc,
            targets,
            predictions,
            show_cam_loc=False,
            show_dist=False):
    fig = plt.figure(dpi=20, figsize=(10, 10))
    params = {
        'legend.fontsize': 'x-large',
    }
    plt.rcParams.update(params)
    plt.title('Linear motion estimation', fontsize=30)
    min_color = 0.5
    max_color = 1.0

    cm_cam = truncate_colormap(cm.get_cmap('Greys'), min_color, max_color)
    cm_gt = truncate_colormap(cm.get_cmap('Purples'), min_color, max_color)
    color_palette = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr',
        'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu',
        'PuBuGn', 'BuGn', 'YlGn'
    ]
    shape_palette = ['^', '*', 'd', 'P', 'p', '1', '8']

    # Plot only valid locations
    valid = (np.sum(targets != 0, axis=1) > 0)
    cam_loc = cam_loc[valid]
    targets = targets[valid]
    predictions['Obs'] = predictions['Obs'][valid]
    predictions['Ref'] = predictions['Ref'][valid[1:]]
    predictions['Prd'] = predictions['Prd'][valid[:-1]]

    cen = np.mean(targets, axis=0)
    var = np.max(np.abs(targets - cen), axis=0)
    var[var < 3.0] = 3.0

    ax = Axes3D(fig)
    ax.set_xlim(cen[0] - var[0], cen[0] + var[0])
    ax.set_ylim(cen[1] - var[1], cen[1] + var[1])
    ax.set_zlim(cen[2] - var[2], cen[2] + var[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.plot(
        targets[:, 0],
        targets[:, 1],
        zs=targets[:, 2],
        color='grey',
        linewidth=2.0,
        label='_nolegend_')
    ax.scatter(
        targets[:, 0],
        targets[:, 1],
        zs=targets[:, 2],
        c=np.linspace(0.0, 1.0, targets.shape[0]),
        cmap=cm_gt,
        marker='o',
        linewidth=4.0,
        label='GT')
    for indx in range(len(targets)):
        ax.text(
            targets[indx, 0],
            targets[indx, 1],
            targets[indx, 2],
            f'GT{indx}',
            size=10,
            zorder=1,
            color='k')

    for key_indx, key in enumerate(predictions):
        preds = predictions[key]
        init_indx = 0 if key == 'Obs' else 1
        cmap_set = color_palette[key_indx + 2]
        mark_set = shape_palette[key_indx + 1]
        cm_in = truncate_colormap(cm.get_cmap(cmap_set), min_color, max_color)
        plt.plot(
            preds[:, 0],
            preds[:, 1],
            zs=preds[:, 2],
            color=cm.get_cmap(cmap_set)(0.5),
            linewidth=1.0,
            label='_nolegend_')
        ax.scatter(
            preds[:, 0],
            preds[:, 1],
            zs=preds[:, 2],
            c=np.linspace(0.0, 1.0, preds.shape[0]),
            cmap=cm_in,
            marker=mark_set,
            linewidth=2.0,
            label=key)
        for indx in range(len(preds)):
            ax.text(
                preds[indx, 0] - var[0] * 0.05,
                preds[indx, 1] - var[1] * 0.05,
                preds[indx, 2] + var[2] * 0.05 * (key_indx - 1),
                f'{key}{indx+init_indx}',
                size=10,
                zorder=1,
                color='k')

    if show_cam_loc:
        plt.plot(
            cam_loc[:, 0],
            cam_loc[:, 1],
            zs=cam_loc[:, 2],
            color='c',
            linewidth=1.0,
            label='_nolegend_')
        ax.scatter(
            cam_loc[:, 0],
            cam_loc[:, 1],
            zs=cam_loc[:, 2],
            c=np.linspace(0.0, 1.0, cam_loc.shape[0]),
            cmap=cm_cam,
            marker=shape_palette[0],
            linewidth=2.0,
            label='CAM')
        ax.text(
            cam_loc[0, 0],
            cam_loc[0, 1],
            cam_loc[0, 2],
            "cam_loc",
            size=10,
            zorder=1,
            color='k')

    plt.legend()
    save_path = osp.join(save_dir, session, f'{epoch}_3D.svg')
    if not osp.isdir(osp.dirname(save_path)):
        os.mkdir(osp.dirname(save_path))
    plt.savefig(save_path, format='svg')
    plt.close()


def plot_segment_and_gt(segm_gt: np.ndarray,
                        segm_pd: np.ndarray,
                        box_pd: np.ndarray,
                        gray_img: np.ndarray = None,
                        range_img: np.ndarray = None,
                        _f: float = 0.3):

    # Generate canvas
    draw_img_gt = np.zeros_like(segm_gt)
    draw_img_pd = np.zeros_like(segm_gt)

    box_pd_int = box_pd.astype(int)
    box_pd_int[:, 0:2][box_pd_int[:, 0:2] < 0] = 0
    box_pd_int[:,
               2:3][box_pd_int[:, 2:3] > segm_gt.shape[1]] = segm_gt.shape[1]
    box_pd_int[:,
               3:4][box_pd_int[:, 3:4] > segm_gt.shape[0]] = segm_gt.shape[0]

    # Iterate through predicted boxes
    for (segm_, bb) in zip(segm_pd, box_pd_int):
        # Get PD visualized
        segm_resize = cv2.resize(
            segm_ * 255, (bb[2] - bb[0], bb[3] - bb[1]),
            interpolation=cv2.INTER_NEAREST)

        draw_img_pd[bb[1]:bb[3], bb[0]:bb[2]] |= segm_resize

        # Crop GT from whole image
        segm_gt_crop = segm_gt[bb[1]:bb[3], bb[0]:bb[2]]
        segm_gt_filter = segm_gt_crop[np.bitwise_and(segm_gt_crop != 0,
                                                     segm_gt_crop != 255)]

        # Check if we have tid after filter
        tid_ = np.median(segm_gt_filter) if len(segm_gt_filter) else -10

        # Get GT visualized
        segm_gt_ = np.zeros_like(segm_gt_crop, dtype=np.uint8)
        segm_gt_[segm_gt_crop == tid_] = 255
        segm_gt_[segm_gt_crop == 255] = 128
        draw_img_gt[bb[1]:bb[3], bb[0]:bb[2]] = segm_gt_

        cv2.rectangle(segm_gt, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 3)
        if gray_img is not None:
            cv2.rectangle(gray_img, (bb[0], bb[1]), (bb[2], bb[3]),
                          (255, 0, 0), 3)

    # Concatenate four canvas into one
    draw_canvas = np.hstack([
        np.vstack([segm_gt, abs(draw_img_gt - draw_img_pd)]),
        np.vstack([draw_img_gt, draw_img_pd])
    ])
    if gray_img is not None and range_img is not None:
        draw_canvas = np.vstack(
            [np.hstack([gray_img, range_img]), draw_canvas])

    while (True):
        cv2.imshow(
            'preview',
            cv2.resize(
                draw_canvas, (0, 0),
                fx=_f,
                fy=_f,
                interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(0) in [ord(' '), 27]:
            break
