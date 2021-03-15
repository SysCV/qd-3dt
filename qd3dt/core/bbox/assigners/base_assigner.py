from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):

    @abstractmethod
    def assign(self,
               bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               gt_depths=None,
               gt_alphas=None,
               gt_rotys=None,
               gt_dims=None,
               gt_2dcs=None,
               ):
        pass
