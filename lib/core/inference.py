from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from ..utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    # The shape of heatmaps_reshaped is[batch_size, num_joints, height*width]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)    # 最大索引
    maxvals = np.amax(heatmaps_reshaped, 2)  # 最大值

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)   # 沿第三维复制扩大两倍
    preds[:, :, 0] = (preds[:, :, 0]) % width            # x_coords
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)  # y_coords

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))  # maxvals大于0，标记为1，否则为0
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    """
    get final predictions(post-processing and transform back to the original image)
    """
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processin
    # 个人理解是消除上下取整带来的影响
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1], hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25  # 取数字正负符号
    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )
    return preds, maxvals
