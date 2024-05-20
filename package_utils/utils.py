#-*- coding: utf-8 -*-
import os
import simplejson as json
from copy import deepcopy
import logging

import cv2
import numpy as np
import torch
from mmcv.utils import get_logger

from losses.losses import _sigmoid


def file_extention(file_path):
    f_name, f_extension = os.path.splitext(file_path)
    return f_name, f_extension


def make_dir(dir_path):
    if not os.path.exists(dir_path):
       os.mkdir(dir_path)


def vis_heatmap(image, heatmaps, file_name):
    hm_h, hm_w = heatmaps.shape[1:]
    masked_image = np.zeros((hm_h, hm_w*heatmaps.shape[0], 3), dtype=np.uint8)

    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        heatmap = np.clip(heatmap*255, 0, 255).astype(np.uint8)
        heatmap = np.squeeze(heatmap)
        
        heatmap_h = heatmap.shape[0]
        heatmap_w = heatmap.shape[1]
        
        resized_image = cv2.resize(image, (int(heatmap_h), int(heatmap_w)))
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        masked_image[:, hm_w*i:hm_w*(i+1), :] = colored_heatmap*0.7 + resized_image*0.3
    cv2.imwrite(file_name, masked_image)


def save_batch_heatmaps(batch_image, 
                        batch_heatmaps, 
                        file_name,
                        normalize=True,
                        batch_cls=None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    batch_cls: ['batch_size, num_joints, 1]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)
    
    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    for i in range(batch_size):
        if batch_image.dim() == 4:
            image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()    
        else:
            image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 3, 0)\
                              .cpu().numpy()\

        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            if image.ndim == 4:
                image = image[j, :, :, :]

            resized_image = cv2.resize(image,
                                       (int(heatmap_width), int(heatmap_height)))

            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            if batch_cls is not None:
                cls = batch_cls[i][j].detach().cpu().numpy()
                colored_heatmap = cv2.putText(colored_heatmap, 
                                              f'Cls Pred: {cls}', 
                                              (heatmap_width*(j+1)-15, 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 
                                              1, 1, 
                                              cv2.LINE_AA)
            masked_image = colored_heatmap*0.7 + resized_image*0.3

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
    cv2.imwrite(file_name, grid_image)


def debugging_panel(debug_cfg, 
                    batch_image, 
                    batch_heatmaps_gt, 
                    batch_heatmaps_pred, 
                    idx, 
                    normalize=True,
                    batch_cls_gt=None,
                    batch_cls_pred=None,
                    split='train'):
    if debug_cfg.save_hm_gt:
        save_batch_heatmaps(batch_image, 
                            batch_heatmaps_gt, 
                            f'samples/{split}_debugs/hm_gt_{idx}.jpg', 
                            normalize=normalize)
    
    if debug_cfg.save_hm_pred:
        batch_heatmaps_pred_ = _sigmoid(batch_heatmaps_pred.clone())
        save_batch_heatmaps(batch_image, 
                            batch_heatmaps_pred_, 
                            f'samples/{split}_debugs/hm_pred_{idx}.jpg',
                            normalize=normalize)


def save_file(data, file_path):
    f_name, f_extention = file_extention(file_path)
    
    if f_extention == '.json':
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f'Data has been saved to --- {file_path}')
    else:
        raise ValueError(f'{f_extention} is not supported now!')


def load_file(file_path):
    f_name, f_extention = file_extention(file_path)
    
    if f_extention == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f'Data has been loaded from --- {file_path}')
    else:
        raise ValueError(f'{f_extention} is not supported now!')

    return data


def draw_landmarks(image, landmarks):
    """This function is to draw facial landmarks into transformed images
    """
    assert landmarks is not None, "Landmarks can not be None!"
    
    img_cp = deepcopy(image)
    
    for i, p in enumerate(landmarks):
        img_cp = cv2.circle(img_cp, (p[0], p[1]), 2, (0, 255, 0), 1)
    
    return img_cp


def draw_most_vul_points(blended_mask):
    """ Detecting and Drawing the most vulnerable points for visualization purpose
    """
    b_mask_cp = deepcopy(blended_mask)
    patches = [[0, 0], [0, 1/2], [1/2, 0], [1/2, 1/2]]
    target_H, target_W = b_mask_cp[..., 0].shape[:2]

    for fr in range(len(patches)):
        p_x1, p_y1 = int(target_W * patches[fr][0]), int(target_H * patches[fr][1])
        p_x2, p_y2 = int(target_W * (patches[fr][0] + 1/2)), int(target_H * (patches[fr][1] + 1/2))

        max_value = b_mask_cp[p_y1:p_y2, p_x1:p_x2, 0].max()
        max_value = max_value if max_value > 0 else 1
        target_mask_ = (b_mask_cp[p_y1:p_y2, p_x1:p_x2, 0] == (max_value)).astype(np.uint8)
        points = np.where(target_mask_ == 1)

        if len(points[0]):
            p = (points[0] + p_y1, points[1] + p_x1)

            for j, i in zip(p[0], p[1]):
                b_mask_cp = cv2.circle(b_mask_cp, (i, j), 2, (255, 0, 0), 1)
    
    return b_mask_cp


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use `get_logger` method in mmcv to get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmpose".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)
