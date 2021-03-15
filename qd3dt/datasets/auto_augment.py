import copy

import numpy as np
from PIL import Image, ImageOps
from PIL.ImageEnhance import Sharpness as PILSharpness
from PIL.ImageFilter import GaussianBlur

_MAX_LEVEL = 10.


def auto_augment(img, gt_bboxes, policies):
    h, w, _ = img.shape
    gt_bboxes = swap_box(gt_bboxes)
    gt_bboxes[:, 0] /= h
    gt_bboxes[:, 2] /= h
    gt_bboxes[:, 1] /= w
    gt_bboxes[:, 3] /= w

    for policy in policies:
        policy = policy.copy()
        policy.pop('share')
        p = eval(policy.pop('type'))(**policy)
        img, gt_bboxes = p(img, gt_bboxes)

    gt_bboxes = swap_box(gt_bboxes)
    h, w, c = img.shape
    gt_bboxes[:, 0] *= w
    gt_bboxes[:, 2] *= w
    gt_bboxes[:, 1] *= h
    gt_bboxes[:, 3] *= h

    return img, gt_bboxes


class Translate:

    def __init__(self, level, prob, replace, axis):
        self.level = level
        self.prob = prob
        self.replace = replace
        self.axis = axis

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        pixels = self.level_to_arg()
        img = self.translate_img(img, pixels, self.replace, self.axis)
        bboxes = self.translate_bbox(bboxes, pixels, self.axis, img.shape[0],
                                     img.shape[1])
        return img, bboxes

    def level_to_arg(self):
        level = (self.level / _MAX_LEVEL) * 250.
        level = random_negative(level)
        return level

    @staticmethod
    def translate_img(img, pixels, replace, axis):
        assert axis in ('x', 'y')
        if axis == 'x':
            trans = (1, 0, pixels, 0, 1, 0)
        else:
            trans = (1, 0, 0, 0, 1, pixels)
        img = Image.fromarray(img)
        img = img.transform(img.size, Image.AFFINE, trans, fillcolor=replace)
        return np.array(img)

    @staticmethod
    def translate_bbox(bbox, pixels, axis, image_height, image_width):
        assert axis in ('x', 'y')
        bbox = bbox.copy()

        min_y = (image_height * bbox[:, 0]).astype(np.int)
        min_x = (image_width * bbox[:, 1]).astype(np.int)
        max_y = (image_height * bbox[:, 2]).astype(np.int)
        max_x = (image_width * bbox[:, 3]).astype(np.int)

        if axis == 'x':
            min_x = np.maximum(0, min_x - pixels)
            max_x = np.minimum(image_width, max_x - pixels)
        else:
            min_y = np.maximum(0, min_y - pixels)
            max_y = np.minimum(image_height, max_y - pixels)

        # Convert bbox back to floats.
        min_y = np.float32(min_y) / np.float32(image_height)
        min_x = np.float32(min_x) / np.float32(image_width)
        max_y = np.float32(max_y) / np.float32(image_height)
        max_x = np.float32(max_x) / np.float32(image_width)

        min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
        min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y,
                                                      max_x)
        return np.stack([min_y, min_x, max_y, max_x], -1)


class TranslateOnlyBBox(Translate):

    def __call__(self, img, bboxes):
        pixels = self.level_to_arg()
        random_inds = np.random.permutation(bboxes.shape[0])

        for idx in random_inds:
            bbox = bboxes[idx]
            if np.random.rand() < self.prob / 3.:
                image_height = np.float32(img.shape[0])
                image_width = np.float32(img.shape[1])
                min_y = np.int32(image_height * bbox[0])
                min_x = np.int32(image_width * bbox[1])
                max_y = np.int32(image_height * bbox[2])
                max_x = np.int32(image_width * bbox[3])
                image_height = np.int32(image_height)
                image_width = np.int32(image_width)

                max_y = min(max_y, image_height - 1)
                max_x = min(max_x, image_width - 1)

                bbox_content = img[min_y:max_y + 1, min_x:max_x + 1, :]
                augmented_bbox_content = self.translate_img(
                    bbox_content, pixels, self.replace, self.axis)
                img[min_y:max_y + 1, min_x:max_x + 1] = augmented_bbox_content

        return img, bboxes

    def level_to_arg(self):
        level = (self.level / _MAX_LEVEL) * 120.
        level = random_negative(level)
        return level


class Cutout:

    def __init__(self, level, prob, replace, ratios, max_cutout):
        self.level = level
        self.prob = prob
        self.replace = replace
        self.ratios = ratios
        self.max_cutout = max_cutout

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        cutout_size = self.level_to_arg()
        ratio = np.random.uniform(self.ratios[0], self.ratios[1])
        img = img.copy()
        h, w, _ = img.shape
        cutout_w = np.int32(min(cutout_size * ratio, w * self.max_cutout))
        cutout_h = np.int32(min(cutout_size, h * self.max_cutout))

        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        x1 = max(0, x - cutout_w)
        y1 = max(0, y - cutout_h)
        x2 = min(w, x + cutout_w)
        y2 = min(h, y + cutout_h)

        img[y1:y2, x1:x2, :] = np.array(
            self.replace)[np.newaxis, np.newaxis, :]
        return img, bboxes

    def level_to_arg(self):
        return int((self.level / _MAX_LEVEL) * 100)


class CutoutOnlyBBox(Cutout):

    def __call__(self, img, bboxes):
        cutout_size = self.level_to_arg()
        random_inds = np.random.permutation(bboxes.shape[0])
        image_height = np.float32(img.shape[0])
        image_width = np.float32(img.shape[1])

        for idx in random_inds:
            bbox = bboxes[idx]
            ratio = np.random.uniform(self.ratios[0], self.ratios[1])
            min_y = np.int32(image_height * bbox[0])
            min_x = np.int32(image_width * bbox[1])
            max_y = np.int32(image_height * bbox[2])
            max_x = np.int32(image_width * bbox[3])

            if np.random.rand() > self.prob or max_x <= min_x or max_y <= min_y:
                continue

            h, w = max_y - min_y, max_x - min_x
            cutout_w = np.int32(min(cutout_size * ratio, w * self.max_cutout))
            cutout_h = np.int32(min(cutout_size, h * self.max_cutout))

            x = np.random.randint(min_x, max_x)
            y = np.random.randint(min_y, max_y)

            x1 = max(min_x, x - cutout_w)
            y1 = max(min_y, y - cutout_h)
            x2 = min(max_x, x + cutout_w)
            y2 = min(max_y, y + cutout_h)

            img[y1:y2, x1:x2, :] = np.array(
                self.replace)[np.newaxis, np.newaxis, :]
        return img, bboxes


class Shear:

    def __init__(self, level, prob, replace, axis):
        self.level = level
        self.prob = prob
        self.replace = replace
        self.axis = axis

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        level = self.level_to_arg()
        if self.axis == 'x':
            trans = (1, level, 0, 0, 1, 0)
        else:
            trans = (1, 0, 0, level, 1, 0)
        image = Image.fromarray(img)
        image = image.transform(
            image.size,
            Image.AFFINE,
            trans,
            resample=Image.NEAREST,
            fillcolor=self.replace)

        bboxes = self.shear_bbox(bboxes, level, self.axis, img.shape[0],
                                 img.shape[1])
        return np.array(image), bboxes

    def level_to_arg(self):
        level = (self.level / _MAX_LEVEL) * 0.3
        level = random_negative(level)
        return level

    def shear_bbox(self, bboxes, level, axis, image_height, image_width):
        new_bboxes = np.zeros_like(bboxes)
        for i in range(bboxes.shape[0]):
            new_bboxes[i] = self._shear_bbox(bboxes[i], level, axis,
                                             image_height, image_width)
        return new_bboxes

    @staticmethod
    def _shear_bbox(bbox, level, axis, image_height, image_width):
        image_height, image_width = (np.float32(image_height),
                                     np.float32(image_width))

        # Change bbox coordinates to be pixels.
        min_y = np.int32(image_height * bbox[0])
        min_x = np.int32(image_width * bbox[1])
        max_y = np.int32(image_height * bbox[2])
        max_x = np.int32(image_width * bbox[3])
        coordinates = np.stack([[min_y, min_x], [min_y, max_x], [max_y, min_x],
                                [max_y, max_x]]).astype(np.float32)

        if axis == 'x':
            translation_matrix = np.stack([[1, 0], [-level,
                                                    1]]).astype(np.float32)
        else:
            translation_matrix = np.stack([[1, -level],
                                           [0, 1]]).astype(np.float32)
        new_coords = np.matmul(translation_matrix,
                               np.transpose(coordinates)).astype(np.int32)

        min_y = np.float32(np.min(new_coords[0, :])) / image_height
        min_x = np.float32(np.min(new_coords[1, :])) / image_width
        max_y = np.float32(np.max(new_coords[0, :])) / image_height
        max_x = np.float32(np.max(new_coords[1, :])) / image_width

        min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
        min_y, min_x, max_y, max_x = _check_bbox_area_single(
            min_y, min_x, max_y, max_x)

        return np.stack([min_y, min_x, max_y, max_x])


class Rotate:

    def __init__(self, level, prob, replace):
        self.level = level
        self.prob = prob
        self.replace = replace

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        degree = self.level_to_arg()
        bboxes = bboxes.copy()
        h, w, _ = img.shape
        img = Image.fromarray(img)
        img = img.rotate(degree, fillcolor=self.replace)

        new_bboxes = np.zeros_like(bboxes)
        for i in range(len(new_bboxes)):
            new_bboxes[i] = self._rotate_bbox(bboxes[i], degree, h, w)

        return np.array(img), new_bboxes

    def level_to_arg(self):
        level = (self.level / _MAX_LEVEL) * 30.
        level = random_negative(level)
        return level

    @staticmethod
    def _rotate_bbox(bbox, degrees, image_height, image_width):
        image_height, image_width = (np.float32(image_height),
                                     np.float32(image_width))

        # Convert from degrees to radians.
        degrees_to_radians = np.pi / 180.0
        radians = degrees * degrees_to_radians

        min_y = -np.int32(image_height * (bbox[0] - 0.5))
        min_x = np.int32(image_width * (bbox[1] - 0.5))
        max_y = -np.int32(image_height * (bbox[2] - 0.5))
        max_x = np.int32(image_width * (bbox[3] - 0.5))
        coordinates = np.stack([[min_y, min_x], [min_y, max_x], [max_y, min_x],
                                [max_y, max_x]]).astype(np.float32)
        rotation_matrix = np.stack([[np.cos(radians),
                                     np.sin(radians)],
                                    [-np.sin(radians),
                                     np.cos(radians)]])
        new_coords = np.matmul(rotation_matrix,
                               np.transpose(coordinates)).astype(np.int32)
        # Find min/max values and convert them back to normalized 0-1 floats.
        min_y = -(np.float32(np.max(new_coords[0, :])) / image_height - 0.5)
        min_x = np.float32(np.min(new_coords[1, :])) / image_width + 0.5
        max_y = -(np.float32(np.min(new_coords[0, :])) / image_height - 0.5)
        max_x = np.float32(np.max(new_coords[1, :])) / image_width + 0.5

        # Clip the bboxes to be sure the fall between [0, 1].
        min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
        min_y, min_x, max_y, max_x = _check_bbox_area_single(
            min_y, min_x, max_y, max_x)

        return np.stack([min_y, min_x, max_y, max_x])


class Blur:

    def __init__(self, level, prob):
        self.level = level
        self.prob = prob

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        factor = self.level_to_arg()
        img = Image.fromarray(img)
        img = img.filter(GaussianBlur(radius=factor))
        return np.array(img), bboxes

    def level_to_arg(self):
        return (self.level / _MAX_LEVEL) * 10


class Color:

    def __init__(self, level, prob):
        self.level = level
        self.prob = prob

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        factor = self.level_to_arg()
        img1 = Image.fromarray(img).convert('L')
        img1 = np.array(img1)
        img1 = np.tile(img1[..., np.newaxis], (1, 1, 3))
        return self.blind(img1, img, factor), bboxes

    def level_to_arg(self):
        return (self.level / _MAX_LEVEL) * 1.8 + 0.1

    @staticmethod
    def blind(img1, img2, factor):
        if factor == 0.0:
            return img1
        if factor == 1.0:
            return img2

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        difference = img2 - img1
        scaled = factor * difference

        tmp = img1 + scaled

        if 0.0 < factor < 1.0:
            return tmp.astype(np.uint8)
        return tmp.clip(min=0.0, max=255.0).astype(np.uint8)


class Sharpness:

    def __init__(self, level, prob):
        self.level = level
        self.prob = prob

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        factor = self.level_to_arg()
        img = Image.fromarray(img)
        img = PILSharpness(img).enhance(factor)
        return np.array(img), bboxes

    def level_to_arg(self):
        return (self.level / _MAX_LEVEL) * 1.8 + 0.1


class Equalize:

    def __init__(self, level, prob):
        self.level = level
        self.prob = prob

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        img = Image.fromarray(img)
        img = ImageOps.equalize(img)
        return np.array(img), bboxes


def random_negative(level):
    if np.random.rand() < 0.5:
        return -level
    return level


def _clip_bbox(min_y, min_x, max_y, max_x):
    min_y = np.clip(min_y, 0.0, 1.0)
    min_x = np.clip(min_x, 0.0, 1.0)
    max_y = np.clip(max_y, 0.0, 1.0)
    max_x = np.clip(max_x, 0.0, 1.0)
    return min_y, min_x, max_y, max_x


def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
    height = max_y - min_y
    width = max_x - min_x

    min_y[height == 0] = np.minimum(min_y[height == 0], 1. - delta)
    max_y[height == 0] = np.maximum(max_y[height == 0], 0. + delta)
    min_x[width == 0] = np.minimum(min_x[width == 0], 1. - delta)
    max_x[width == 0] = np.maximum(max_x[width == 0], 0. + delta)
    return min_y, min_x, max_y, max_x


def _check_bbox_area_single(min_y, min_x, max_y, max_x, delta=0.05):
    height = max_y - min_y
    width = max_x - min_x

    if height == 0:
        min_y = np.minimum(min_y, 1. - delta)
        max_y = np.maximum(max_y, 0. + delta)
    if width == 0:
        min_x = np.minimum(min_x, 1. - delta)
        max_x = np.maximum(max_x, 0. + delta)
    return min_y, min_x, max_y, max_x


def swap_box(bboxes):
    new_bboxes = np.zeros_like(bboxes)
    new_bboxes[:, 0::2] = bboxes[:, 1::2]
    new_bboxes[:, 1::2] = bboxes[:, 0::2]
    return new_bboxes
