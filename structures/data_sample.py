#-*- coding: utf-8 -*-
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmengine.structures import BaseDataElement, InstanceData, PixelData


class DataSample(BaseDataElement):
    """The base data structure of DEFADET that is used as the interface between
    modules.

    The attributes of ``DataSample`` includes:
        - ``pred_instances``(InstanceData): Instances with keypoint
            predictions
        - ``gt_fields``(PixelData): Ground truth of spatial distribution
            annotations like keypoint heatmaps and part affine fields (PAF)
        - ``pred_fields``(PixelData): Predictions of spatial distributions

    Examples:
        >>> import torch
        >>> from mmengine.structures import InstanceData, PixelData
        >>> from structures import DataSample

        >>> fake_meta = dict(img_shape=(800, 1216),
        ...                  crop_size=(256, 192),
        ...                  heatmap_size=(64, 48))
        >>> gt_instances = InstanceData()
        >>> gt_instances.bboxes = torch.rand((1, 4))
        >>> gt_instances.keypoints = torch.rand((1, 17, 2))
        >>> gt_instances.keypoints_visible = torch.rand((1, 17, 1))
        >>> gt_fields = PixelData()
        >>> gt_fields.heatmaps = torch.rand((17, 64, 48))

        >>> data_sample = PoseDataSample(gt_instances=gt_instances,
        ...                              gt_fields=gt_fields,
        ...                              metainfo=pose_meta)
        >>> assert 'img_shape' in data_sample
        >>> len(data_sample.gt_intances)
        1
    """
    @property
    def gt_instance_labels(self) -> InstanceData:
        return self._gt_instance_labels

    @gt_instance_labels.setter
    def gt_instance_labels(self, value: InstanceData):
        self.set_field(value, '_gt_instance_labels', dtype=InstanceData)

    @gt_instance_labels.deleter
    def gt_instance_labels(self):
        del self._gt_instance_labels

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances
