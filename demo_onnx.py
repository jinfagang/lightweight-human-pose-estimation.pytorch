import argparse
import enum
import cv2
import numpy as np
from modules.pose import Pose, track_poses
from val import normalize
from alfred.utils.timer import ATimer
from wanwu.core.backends.ort import ORTWrapper
import math

BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(
                self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale,
                            fy=scale, interpolation=cv2.INTER_LINEAR)
    print(scaled_img.shape)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(
        scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    inp_img = np.expand_dims(padded_img.transpose((2, 0, 1)), axis=0)
    print(inp_img.shape)
    stages_output = net.infer(inp_img)
    print(stages_output)
    heatmaps = stages_output[0]
    pafs = stages_output[1]
    return heatmaps, pafs, scale, pad


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def infer_fast2(net, img, net_input_height_size, stride, upsample_ratio, cpu,
                pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    # net_in_height = 256
    # net_in_width = 256
    net_in_height = 256
    net_in_width = 288

    scale = min(net_in_height / height, net_in_width/width)

    scaled_img = cv2.resize(
        img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # scaled_img = normalize(scaled_img, img_mean, img_scale)
    s_h, s_w, _ = scaled_img.shape
    in_img = np.ones([net_in_height, net_in_width, 3]).astype(np.uint8) * 128
    top = (net_in_height - s_h) // 2
    left = (net_in_width - s_w) // 2
    in_img[top: top + s_h, left: left + s_w] = scaled_img

    # cv2.imshow('aa', in_img)
    # cv2.waitKey(0)

    in_img = normalize(in_img, img_mean, img_scale, )
    inp_img = np.expand_dims(in_img.transpose((2, 0, 1)), axis=0)
    print(inp_img.shape)
    stages_output = net.infer(inp_img)
    # print(stages_output)
    heatmaps = stages_output['stage_1_output_1_heatmaps'].squeeze(0)
    pafs = stages_output['stage_1_output_0_pafs'].squeeze(0)
    print(heatmaps.shape)
    print(pafs.shape)
    # now heatmaps are coords, like [1, 19, 3]
    return heatmaps, pafs, scale, [top, left]


def connections_nms(a_idx, b_idx, affinity_scores):
    # From all retrieved connections that share the same starting/ending keypoints leave only the top-scoring ones.
    order = affinity_scores.argsort()[::-1]
    affinity_scores = affinity_scores[order]
    a_idx = a_idx[order]
    b_idx = b_idx[order]
    idx = []
    has_kpt_a = set()
    has_kpt_b = set()
    for t, (i, j) in enumerate(zip(a_idx, b_idx)):
        if i not in has_kpt_a and j not in has_kpt_b:
            idx.append(t)
            has_kpt_a.add(i)
            has_kpt_b.add(j)
    idx = np.asarray(idx, dtype=np.int32)
    return a_idx[idx], b_idx[idx], affinity_scores[idx]


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05):
    pose_entries = []
    all_keypoints = np.array(
        [item for sublist in all_keypoints_by_type for item in sublist])
    points_per_limb = 10
    grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)
    all_keypoints_by_type = [np.array(keypoints, np.float32)
                             for keypoints in all_keypoints_by_type]

    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        n = len(kpts_a)
        m = len(kpts_b)
        if n == 0 or m == 0:
            continue

        # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
        a = kpts_a[:, :2]
        a = np.broadcast_to(a[None], (m, n, 2))
        b = kpts_b[:, :2]
        vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

        # Sample points along every candidate limb vector.
        steps = (1 / (points_per_limb - 1) * vec_raw)
        points = steps * grid + a.reshape(-1, 1, 2)
        points = points.round().astype(dtype=np.int32)
        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        # Compute affinity score between candidate limb vectors and part affinity field.
        field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
        vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
        vec = vec_raw / (vec_norm + 1e-6)
        affinity_scores = (field * vec).sum(-1).reshape(-1, points_per_limb)
        valid_affinity_scores = affinity_scores > min_paf_score
        valid_num = valid_affinity_scores.sum(1)
        affinity_scores = (affinity_scores *
                           valid_affinity_scores).sum(1) / (valid_num + 1e-6)
        success_ratio = valid_num / points_per_limb

        # Get a list of limbs according to the obtained affinity score.
        valid_limbs = np.where(np.logical_and(
            affinity_scores > 0, success_ratio > 0.8))[0]
        if len(valid_limbs) == 0:
            continue
        b_idx, a_idx = np.divmod(valid_limbs, n)
        affinity_scores = affinity_scores[valid_limbs]

        # Suppress incompatible connections.
        a_idx, b_idx, affinity_scores = connections_nms(
            a_idx, b_idx, affinity_scores)
        connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                               kpts_b[b_idx, 3].astype(np.int32),
                               affinity_scores))
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size)
                            * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(
                    all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i]
                                                             [1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i]
                                            [0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


def run_demo(image_provider, height_size, cpu, track, smooth):

    onnx_model = ORTWrapper('human-pose-estimation.onnx')

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        with ATimer('onnx'):
            heatmaps, pafs, scale, pad = infer_fast2(
                onnx_model, img, height_size, stride, upsample_ratio, cpu)

        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            all_keypoints_by_type.append(
                np.array([np.append(heatmaps[kpt_idx], kpt_idx)]))

        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type, pafs)

        # h, w
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('-m', '--onnx_model', type=str,
                        default='human-pose-estimation.onnx', help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256,
                        help='network input layer height size')
    parser.add_argument('--video', type=str, default='',
                        help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='',
                        help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true',
                        help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1,
                        help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1,
                        help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(frame_provider, args.height_size,
             args.cpu, args.track, args.smooth)
