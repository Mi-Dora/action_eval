from sklearn.cluster import KMeans
import numpy as np
import cv2


# def seg_skeleton(pose_list, segments):
#     pose_segments = []
#     for seg in segments:


def get_seg_interval(static_idx, angle_vec_arr, num_cluster=2):
    static_angle_vec_arr = extract_static_angle_vec(static_idx, angle_vec_arr)
    print(static_idx)
    cluster_center, labels = angle_vec_cluster(static_angle_vec_arr, num_cluster)
    print(labels)
    dis = feature_distance(static_angle_vec_arr, cluster_center, labels)
    last_label = -1
    last_pos = -1
    mask = np.ones_like(labels)
    for i in range(len(static_idx)):
        label = labels[i]
        if label == last_label:
            if dis[i] < dis[last_pos]:
                mask[last_pos] = 0
                last_pos = i
            else:
                mask[i] = 0

        else:
            last_label = label
            last_pos = i
    print(mask)
    seg_label = -1
    mid_label = -1
    for i in range(len(static_idx)):
        if mask[i] == 1 and seg_label == -1:
            seg_label = labels[i]
            continue
        if mask[i] == 1 and labels[i] != seg_label:
            mid_label = labels[i]
            break

    start = -1
    mid = -1
    segments = []
    for i in range(len(static_idx)):
        if mask[i] == 0:
            continue
        if labels[i] == seg_label and start == -1:
            start = static_idx[i]
            continue
        if labels[i] == mid_label and start != -1:
            mid = static_idx[i]
            continue
        if labels[i] == seg_label and mid != -1:
            end = static_idx[i]
            segments.append([start, mid, end])
            start = end
            mid = -1
    return segments


def extract_static_angle_vec(static_idx, angle_vec_arr):
    feature_len = angle_vec_arr.shape[1]
    num_static = len(static_idx)
    extracted_arr = np.zeros((num_static, feature_len))
    for i, idx in enumerate(static_idx):
        extracted_arr[i, :] = angle_vec_arr[idx, :]
    return extracted_arr


def angle_vec_cluster(static_angle_vec_arr, num_cluster=2):
    clt = KMeans(n_clusters=num_cluster)

    clt.fit(static_angle_vec_arr)
    cluster_center = clt.cluster_centers_
    labels = clt.labels_
    return cluster_center, labels


def feature_distance(static_angle_vec_arr, cluster_center, labels):
    center_arr = np.zeros_like(static_angle_vec_arr)
    for i, label in enumerate(labels):
        center_arr[i, :] = cluster_center[label, :]
    dis = np.sqrt(((static_angle_vec_arr - center_arr) ** 2).sum(axis=1))
    print(dis)
    return dis

