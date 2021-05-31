from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.pyplot as plt
from src.tools.pose_angle import feature_len
from src.tools.utils import plot_circle


# def seg_skeleton(pose_list, segments):
#     pose_segments = []
#     for seg in segments:


def get_seg_interval(static_idx, angle_vec_arr, num_cluster=2, mask=None, video_name=''):
    if mask is None:
        mask = np.ones(feature_len)
    static_angle_vec_arr = extract_static_angle_vec(static_idx, angle_vec_arr)
    # print(static_idx)
    cluster_center, labels = angle_vec_cluster(static_angle_vec_arr, num_cluster)
    print(labels)
    dis, rev_dis, center_dis = feature_distance(static_angle_vec_arr, cluster_center, labels)
    last_label = -1
    last_pos = -1
    mask = np.ones_like(labels)
    for i in range(len(static_idx)):
        label = labels[i]
        if label == last_label:
            if rev_dis[i] < 0.7*center_dis[0, 1]:
                labels[i] = 1 - labels[i]
                last_pos = i
                last_label = labels[i]
            elif dis[i] < dis[last_pos]:
                mask[last_pos] = 0
                last_pos = i
            else:
                mask[i] = 0
        else:
            last_label = label
            last_pos = i
    # print(mask)
    get_processed_idx_label(static_idx, labels, mask)
    save_path = os.path.split(os.path.realpath(__file__))[0] + '/../../plots/clustering/' + video_name.split('.')[0] + '.png'
    plot_cluster_pca(static_angle_vec_arr, cluster_center, labels, save_path)
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
    dispersion = 1 - np.exp(-dis.sum()/dis.size)
    return segments, dispersion


def get_processed_idx_label(idxs, labels, mask):
    new_idxs = []
    new_labels = []
    for i, v in enumerate(mask):
        if v == 1:
            new_idxs.append(idxs[i])
            new_labels.append(labels[i])
    print('\nFinal key frame:')
    print('"idx":', end='')
    print(new_idxs, end='')
    print(',\n"label":', end='')
    print(new_labels)
    print('\n')


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
    rev_center_arr = np.zeros_like(static_angle_vec_arr)
    center_dis = np.zeros((cluster_center.shape[0], cluster_center.shape[0]))
    center_dis[0, 1] = center_dis[1, 0] = np.sqrt(((cluster_center[0] - cluster_center[1]) ** 2).sum())
    for i, label in enumerate(labels):
        center_arr[i, :] = cluster_center[label, :]
        rev_center_arr[i, :] = cluster_center[1-label, :]
    dis = np.sqrt(((static_angle_vec_arr - center_arr) ** 2).sum(axis=1))
    rev_dis = np.sqrt(((static_angle_vec_arr - rev_center_arr) ** 2).sum(axis=1))
    print('feature distance is:')
    print(dis)
    return dis, rev_dis, center_dis


def plot_cluster_pca(static_angle_vecs, cluster_centers, labels, save_path='./tmp.png'):
    color = ['r', 'b']
    colors = []
    for label in labels:
        colors.append(color[label])
    vecs = np.concatenate((cluster_centers, static_angle_vecs), axis=0)
    # meanVals = np.mean(vecs, axis=0)
    estimator = PCA(n_components=2)
    pca_vecs = estimator.fit_transform(vecs)
    # pca_vecs = estimator.inverse_transform(pca_vecs)
    # pca_static_angle_vecs = estimator.fit_transform(static_angle_vecs)
    # a = estimator.mean_
    # meanVals = np.mean(pca_vecs, axis=0)
    pca_centers = pca_vecs[:cluster_centers.shape[0], :]
    pca_static_angle_vecs = pca_vecs[cluster_centers.shape[0]:, :]

    dis, _, _ = feature_distance(pca_static_angle_vecs, pca_centers, labels)
    r1 = np.sum(dis[labels == 0]) / dis[labels == 0].size
    r2 = np.sum(dis[labels == 1]) / dis[labels == 1].size
    plt.figure(figsize=(5, 5))
    plt.scatter(pca_static_angle_vecs[:, 0], pca_static_angle_vecs[:, 1], c=colors, s=25)
    plt.scatter(pca_centers[:, 0], pca_centers[:, 1], c='#000000', s=50)
    xc1 = pca_centers[0, 0]
    yc1 = pca_centers[0, 1]
    xc2 = pca_centers[1, 0]
    yc2 = pca_centers[1, 1]
    plot_circle(plt, xc1, yc1, r1)
    plot_circle(plt, xc2, yc2, r2)
    k = - (xc1-xc2)/(yc1-yc2)
    xm = (xc1+xc2)/2
    ym = (yc1+yc2)/2
    b = ym - k*xm
    x = np.array([-1.5*np.pi, 1.5*np.pi])
    yy = x*k+b
    plt.plot(x, yy, c='orange')

    x_max = pca_static_angle_vecs[:, 0].max()
    x_min = pca_static_angle_vecs[:, 0].min()
    y_max = pca_static_angle_vecs[:, 1].max()
    y_min = pca_static_angle_vecs[:, 1].min()
    print(x_max, y_max, x_min, y_min)
    # plt.xlim([min(x_min, y_min)-0.1, max(x_max, y_max)+0.1])
    # plt.ylim([min(x_min, y_min)-0.1, max(x_max, y_max)+0.1])
    plt.xlim([-1.5*np.pi, 1.5*np.pi])
    plt.ylim([-1.5*np.pi, 1.5*np.pi])
    plt.title('Clustering')
    plt.savefig(save_path)
    plt.clf()

