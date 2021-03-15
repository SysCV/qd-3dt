import numpy as np
import mmcv
import cv2
import random
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.use('Agg')


def random_color(seed):
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color


def imshow_bboxes_w_ids(img, bboxes, ids, font_scale=0.5, out_file=None):
    assert bboxes.ndim == 2
    assert ids.ndim == 1
    assert bboxes.shape[0] == ids.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    if isinstance(img, str):
        img = plt.imread(img)
    else:
        img = mmcv.bgr2rgb(img)
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.autoscale(False)
    plt.subplots_adjust(
        top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    for bbox, id in zip(bboxes, ids):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        w = bbox_int[2] - bbox_int[0] + 1
        h = bbox_int[3] - bbox_int[1] + 1
        color = random_color(id)
        plt.gca().add_patch(
            Rectangle(left_top, w, h, edgecolor=color, facecolor='none'))
        label_text = '{}'.format(int(id))
        bg_height = 12
        bg_width = 10
        bg_width = len(label_text) * bg_width
        plt.gca().add_patch(
            Rectangle((left_top[0], left_top[1] - bg_height),
                      bg_width,
                      bg_height,
                      edgecolor=color,
                      facecolor=color))
        plt.text(left_top[0] - 1, left_top[1], label_text, fontsize=5)

    if out_file is not None:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)

    plt.clf()
    return img


def imshow_3d_tracklets(img,
                        bboxes,
                        labels,
                        ids=None,
                        depths=None,
                        cen_2d=None,
                        thickness=2,
                        font_scale=0.4,
                        show=False,
                        win_name='',
                        color=None,
                        out_file=None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    if depths is not None:
        assert bboxes.shape[0] == depths.shape[0]
    if ids is not None:
        assert bboxes.shape[0] == ids.shape[0]

    if isinstance(img, str):
        img = cv2.imread(img)

    for indx, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2, _ = bbox.astype(np.int32)
        if color is not None:
            bbox_color = mmcv.color_val(color)
        elif ids is not None:
            bbox_color = random_color(ids[indx])
            bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
        else:
            bbox_color = mmcv.color_val('green')

        if ids is not None:
            info_text = f'T{int(ids[indx]):03d}'
        else:
            info_text = f'D000'

        if depths is not None:
            info_text += f'_{int(depths[indx]):03d}m'

        img[y1:y1+12, x1:x1+80, :] = bbox_color
        cv2.putText(
            img,
            info_text, (x1, y1+10),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=mmcv.color_val('black'))

        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        if bbox[-1] < 0:
            bbox[-1] = np.nan
        label_text = '{:.02f}'.format(bbox[-1])
        img[y1-12:y1, x1:x1+30, :] = bbox_color
        cv2.putText(
            img,
            label_text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=mmcv.color_val('black'))

    if show:
        cv2.imshow(win_name, img)
    if out_file is not None:
        cv2.imwrite(out_file, img)

    return img
