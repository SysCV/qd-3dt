import cv2
import mmcv
import random
import matplotlib
import numpy as np
import seaborn as sns
from PIL import ImageColor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mmcv.image import imread, imwrite
from .color import color_val
import platform
if platform.system() == 'Linux':
    matplotlib.use('Agg')


def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.moveWindow(win_name, 0, 0)
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None,
                      embed_score=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    if isinstance(img, str):
        img = imread(img)
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    i = 0
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else '{}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        if embed_score is None:
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        else:
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[3] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
            cv2.putText(img, '{:.02f}'.format(embed_score[i]),
                        (bbox_int[0], bbox_int[3] + 10),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        i += 1

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def random_color(seed):
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color


def imshow_bboxes_w_ids(img,
                        bboxes,
                        ids,
                        thickness=1,
                        font_scale=0.5,
                        out_file=None):
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
            Rectangle(
                left_top, w, h, thickness, edgecolor=color, facecolor='none'))
        label_text = '{}'.format(int(id))
        bg_height = 15
        bg_width = 12
        bg_width = len(label_text) * bg_width
        plt.gca().add_patch(
            Rectangle((left_top[0], left_top[1] - bg_height),
                      bg_width,
                      bg_height,
                      thickness,
                      edgecolor=color,
                      facecolor=color))
        plt.text(left_top[0] - 1, left_top[1], label_text, fontsize=5)

    if out_file is not None:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.clf()
    return img


def imshow_tracklets(img,
                     bboxes,
                     labels=None,
                     ids=None,
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
    if isinstance(img, str):
        img = imread(img)
    i = 0
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2, _ = bbox.astype(np.int32)
        if ids is not None:
            if color is None:
                bbox_color = random_color(ids[i])
                bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
            else:
                bbox_color = mmcv.color_val(color)
            img[y1:y1 + 12, x1:x1 + 20, :] = bbox_color
            cv2.putText(
                img,
                str(ids[i]), (x1, y1 + 10),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                color=color_val('black'))
        else:
            if color is None:
                bbox_color = color_val('green')
            else:
                bbox_color = mmcv.color_val(color)

        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        if bbox[-1] < 0:
            bbox[-1] = np.nan
        label_text = '{:.02f}'.format(bbox[-1])
        img[y1 - 12:y1, x1:x1 + 30, :] = bbox_color
        cv2.putText(
            img,
            label_text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=color_val('black'))

        i += 1

    if show:
        imshow(img, win_name)
    if out_file is not None:
        imwrite(img, out_file)

    return img
