import numpy as np
import warnings
import cv2
import torch


def topdownhead_decode_heatmaps_without_cs(output):
    """Decode keypoints from heatmaps.

    Args:
        img_metas (list(dict)): Information about data augmentation
            By default this includes:
            - "image_file: path to the image file
            - "center": center of the bbox
            - "scale": scale of the bbox
            - "rotation": rotation of the bbox
            - "bbox_score": score of bbox
        output (np.ndarray[N, K, H, W]): model predicted heatmaps.
    """
    batch_size = output.shape[0]
    preds, maxvals = keypoints_from_heatmaps_without_cs(output)

    all_preds = torch.zeros(
        (batch_size, preds.shape[1], 3), dtype=torch.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    return all_preds


def keypoints_from_heatmaps_without_cs(
    heatmaps,
    unbiased=False,
    post_process="default",
    kernel=11,
    valid_radius_factor=0.0546875,
    use_udp=False,
    target_type="GaussianHeatmap",
):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        batch size: N
        num keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, "megvii"]
    if post_process in ["megvii", "unbiased"]:
        assert kernel > 0
    if use_udp:
        assert not post_process == "megvii"

    # normalize configs
    if post_process is False:
        warnings.warn(
            "post_process=False is deprecated, " "please use post_process=None instead",
            DeprecationWarning,
        )
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                "post_process=True, unbiased=True is deprecated,"
                " please use post_process='unbiased' instead",
                DeprecationWarning,
            )
            post_process = "unbiased"
        else:
            warnings.warn(
                "post_process=True, unbiased=False is deprecated, "
                "please use post_process='default' instead",
                DeprecationWarning,
            )
            post_process = "default"
    elif post_process == "default":
        if unbiased is True:
            warnings.warn(
                "unbiased=True is deprecated, please use "
                "post_process='unbiased' instead",
                DeprecationWarning,
            )
            post_process = "unbiased"

    # start processing
    if post_process == "megvii":
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == "GaussianHeatMap".lower():
            preds, maxvals = _get_max_preds_tensor(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == "CombinedTarget".lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds_tensor(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError(
                "target_type should be either " "'GaussianHeatmap' or 'CombinedTarget'"
            )
    else:
        preds, maxvals = _get_max_preds_tensor(heatmaps)
        if post_process == "unbiased":  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(np.maximum(
                _gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            pass
            # add +/-0.25 shift to the predicted locations for higher acc.
            # this is default behavior
            # for n in range(N):
            #     for k in range(K):
            #         heatmap = heatmaps[n][k]
            #         px = int(preds[n][k][0])
            #         py = int(preds[n][k][1])
            #         if 1 < px < W - 1 and 1 < py < H - 1:
            #             diff = torch.tensor(
            #                 [
            #                     heatmap[py][px + 1] - heatmap[py][px - 1],
            #                     heatmap[py + 1][px] - heatmap[py - 1][px],
            #                 ]
            #             )
            #             preds[n][k] += torch.sign(diff) * 0.25
            #             if post_process == "megvii":
            #                 preds[n][k] += 0.5
    return preds, maxvals


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        heatmap height: H
        heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 *
                      heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1]
            - heatmap[py - 1][px + 1]
            - heatmap[py + 1][px - 1]
            + heatmap[py - 1][px - 1]
        )
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 *
            heatmap[py][px] + heatmap[py - 2 * 1][px]
        )
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        batch size: B
        num keypoints: K
        num persons: N
        height of heatmaps: H
        width of heatmaps: W
        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        res (np.ndarray[N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert B == 1 or B == N
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)
    batch_heatmaps = np.transpose(
        batch_heatmaps, (2, 3, 0, 1)).reshape(H, W, -1)
    batch_heatmaps_pad = cv2.copyMakeBorder(
        batch_heatmaps, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT
    )
    batch_heatmaps_pad = np.transpose(
        batch_heatmaps_pad.reshape(H + 2, W + 2, B, K), (2, 3, 0, 1)
    ).flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray[N, K, H, W]: Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width +
                          2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def _get_max_preds_tensor(heatmaps):
    assert isinstance(
        heatmaps, torch.Tensor), "heatmaps should be torch.tensor for onnx export"
    assert heatmaps.ndim == 4, "batch_images should be 4-ndim"

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
    maxvals = maxvals.unsqueeze(-1)
    idx = idx.unsqueeze(-1)

    idx_repeated = idx.repeat(1, 1, 2).to(torch.float32)
    preds = idx_repeated.clone()
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    a = idx_repeated > 0.0
    preds = torch.where(a, preds, torch.tensor(-1.0).to(torch.float32))
    return preds, maxvals
