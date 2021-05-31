import cv2
import os
import sys
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import tqdm
from src.tools.video_preprocess import gen_video_array
from src.tools. exponential_smooth import plotExponentialSmoothing, plotDoubleExponentialSmoothing, double_exponential_smoothing


def get_static_frames_idx(video_array, interval=1, DEBUG=False, video_name='', smooth=True):
    name = os.path.basename(video_name).split('.')[0] + '.png'
    save_path = os.path.split(os.path.realpath(__file__))[0] + '/../../plots/exp_contrast/' + name
    mov_pix_vec = get_diff_pix_vec(video_array, interval=interval, DEBUG=DEBUG)
    if smooth:
        mov_pix_vec = double_exponential_smoothing(mov_pix_vec, alpha=0.3, beta=0.3, save_path=save_path)
    is_static = True
    first = -1
    static_frames_idx = []
    thresh = mov_pix_vec.max() * 0.2
    begin = -1
    for i, n in enumerate(mov_pix_vec):
        if first == -1:
            if n > 0.5 * thresh and is_static is True:
                first = i
                static_frames_idx.append(int(i))
        else:
            if n < thresh and is_static is False:
                begin = i
                is_static = True
            elif n > thresh and is_static is True:
                is_static = False
                end = i
                if begin != -1:
                    sliced = mov_pix_vec[begin:end]
                    arg = np.argsort(sliced)
                    static_frames_idx.append(int(arg[0])+begin)
                    begin = -1
    if is_static is True:
        for i in range(mov_pix_vec.size - 1, 0, -1):
            if mov_pix_vec[i] > 0.5 * thresh:
                static_frames_idx.append(int(i))
                break

    plot_move_pix(mov_pix_vec, static_frames_idx,
                  os.path.split(os.path.realpath(__file__))[0] + '/../../plots/mov_pix/' + name)
    return static_frames_idx


def get_diff_pix_vec(video_array, interval=1, DEBUG=False):

    num_frames = video_array.shape[0]
    pbar = tqdm.tqdm(total=num_frames)
    # 初始化当前帧的前两帧
    last_frame1 = None
    last_frame2 = None
    frameDelta1 = None
    move_pix_list = []
    cnt = 0
    # 遍历视频的每一帧
    for i in range(num_frames):
        # 读取下一帧
        frame = video_array[i]
        cnt += 1
        if cnt % interval != 0:
            continue

        # 如果第一二帧是None，对其进行初始化,计算第一二帧的不同
        if last_frame2 is None:
            if last_frame1 is None:
                last_frame1 = frame
            else:
                last_frame2 = frame
                frameDelta1 = cv2.absdiff(last_frame1, last_frame2)  # 帧差一
            continue

        # 计算当前帧和前帧的不同,计算三帧差分
        frameDelta2 = cv2.absdiff(last_frame2, frame)  # 帧差二
        thresh = cv2.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算

        # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一
        last_frame1 = last_frame2
        last_frame2 = frame.copy()
        frameDelta1 = frameDelta2

        # 结果转为灰度图
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        if DEBUG:
            cv2.imshow('Frame Differential', thresh)
            cv2.waitKey(10)
        # 图像二值化
        thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
        move_pix_list.append(thresh.sum() / 255)
        pbar.update(interval)
    pbar.close()

    return np.float64(move_pix_list)


# def get_diff_pix_vec(video_file, interval=1, DEBUG=False):
#     video = cv2.VideoCapture(video_file)
#     num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
#     pbar = tqdm.tqdm(total=num_frames)
#     # 初始化当前帧的前两帧
#     last_frame1 = None
#     last_frame2 = None
#     frameDelta1 = None
#     move_pix_list = []
#     cnt = 0
#     # 遍历视频的每一帧
#     while video.isOpened():
#         # 读取下一帧
#         (ret, frame) = video.read()
#         cnt += 1
#         if cnt % interval != 0:
#             continue
#         # 如果不能抓取到一帧，说明我们到了视频的结尾
#         if not ret:
#             break
#
#         # 如果第一二帧是None，对其进行初始化,计算第一二帧的不同
#         if last_frame2 is None:
#             if last_frame1 is None:
#                 last_frame1 = frame
#             else:
#                 last_frame2 = frame
#                 frameDelta1 = cv2.absdiff(last_frame1, last_frame2)  # 帧差一
#             continue
#
#         # 计算当前帧和前帧的不同,计算三帧差分
#         frameDelta2 = cv2.absdiff(last_frame2, frame)  # 帧差二
#         thresh = cv2.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算
#
#         # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一
#         last_frame1 = last_frame2
#         last_frame2 = frame.copy()
#         frameDelta1 = frameDelta2
#
#         # 结果转为灰度图
#         thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
#         if DEBUG:
#             cv2.imshow('Frame Differential', thresh)
#             cv2.waitKey(10)
#         # 图像二值化
#         thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
#         move_pix_list.append(thresh.sum() / 255)
#         pbar.update(1)
#     pbar.close()
#
#     return np.float64(move_pix_list)


def plot_move_pix(mov_pix_vec, static_frames_idx, save_path='tmp.png'):
    plt.title('Moving-Pixel Sequence')
    plt.bar(range(len(mov_pix_vec)), mov_pix_vec)
    for idx in static_frames_idx:
        plt.vlines(idx, 0, mov_pix_vec.max(), colors="r", linestyles="dashed")
    plt.xlabel('Frame ID')
    plt.ylabel('Number of moving pixel')
    plt.savefig(save_path)
    plt.clf()


def key_frame_extract(video_dir):
    video_list = []
    if os.path.isdir(video_dir):
        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_list.append(os.path.join(root, file))
    else:
        video_list.append(video_dir)
    for video_file in video_list:
        print(video_file)
        video_array = gen_video_array(video_file, short_edge=512)
        cat_keyframe = None
        idxs = get_static_frames_idx(video_array, DEBUG=False, video_name=video_file)
        print(video_file)
        print(idxs)
        for i, idx in enumerate(idxs):
            if i == 0:
                cat_keyframe = video_array[idx]
            else:
                cat_keyframe = np.concatenate((cat_keyframe, video_array[idx]), axis=1)
        print(video_file[:-4] + '.png\n')
        cv2.imwrite(video_file[:-4] + '.png', cat_keyframe)


def compute_eval(extracted, gt, thresh=0.1):
    idx_vec = np.zeros(gt.size)
    match = -np.ones(gt.size)
    for i, idx in enumerate(extracted):
        idx_vec[:] = idx
        err = abs(idx_vec - gt)
        pos = np.argmin(err)
        if pos == 0:
            interval = (gt[1] - gt[0]) * 2
        elif pos == gt.size - 1:
            interval = (gt[pos] - gt[pos-1]) * 2
        else:
            interval = gt[pos + 1] - gt[pos - 1]
        if err[pos] < thresh * interval:
            if match[pos] == -1 or match[pos] > err[pos]:
                match[pos] = err[pos]
    TP = gt.size - np.sum(match == -1)
    recall = TP / gt.size  # TP/TP+FN
    precision = TP / extracted.size  # TP/TP+FP
    return TP, recall, precision


def eval_keyframe(label_file, video_path, thresh=0.3, smooth=False):
    print("Threshold = {}".format(thresh))
    f = open(label_file, 'r')
    content = f.read()
    keyframe_label = json.loads(content)
    video_names = keyframe_label.keys()
    tt_gt = 0
    tt_ex = 0
    tt_tp = 0
    json_dict = {}
    for name in video_names:
        gt = keyframe_label[name]
        gt = np.array(gt)
        video_file = os.path.join(video_path, name)
        video_array = gen_video_array(video_file, short_edge=1080)
        extracted = get_static_frames_idx(video_array, DEBUG=False, video_name=video_file, smooth=smooth)
        json_dict[name] = extracted
        extracted = np.array(extracted)
    with open("key_frame_unsmooth.json", "w") as f:
        json_str = json.dumps(json_dict, indent=4)
        f.write(json_str)
    #     TP, recall, precision = compute_eval(extracted, gt, thresh)
    #     print("{}: recall={}, precision={}".format(name, recall, precision))
    #     tt_gt += gt.size
    #     tt_ex += extracted.size
    #     tt_tp += TP
    # tt_recall = tt_tp / tt_gt
    # tt_precision = tt_tp / tt_ex
    # print("Total: recall={}, precision={}".format(tt_recall, tt_precision))
    # return tt_recall, tt_precision


def plot_recall_precision():
    plt.figure(figsize=(6, 4))
    plt.plot(threshes, recalls, marker='^', label='Recall')
    plt.plot(threshes, recalls_wo_smooth, marker='o', ls="dashed", label='Recall w/o smooth')
    # plt.title('Period-Similarity Bar')
    plt.legend(loc='upper left')
    plt.xlabel('Thresh')
    plt.ylabel('Recall')
    plt.savefig('../../plots/eval/recall.png')
    plt.show()
    plt.clf()

    plt.figure(figsize=(6, 4))
    plt.plot(threshes, precisions, marker='^', label='Precision')
    plt.plot(threshes, precisions_wo_smooth, marker='o', ls="dashed", label='Precision w/o smooth')
    # plt.title('Period-Similarity Bar')
    plt.legend(loc='upper left')
    plt.xlabel('Thresh')
    plt.ylabel('Precision')
    plt.savefig('../../plots/eval/precision.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    label_file = "../../eval/keyframe_label.json"
    video_path = '../../data/seg_sample'
    begin = time.time()
    threshes = np.arange(0.05, 0.25, 0.05)
    recalls = []
    recalls_wo_smooth = []
    precisions = []
    precisions_wo_smooth = []
    # for thresh in threshes:
    eval_keyframe(label_file, video_path, 0.1)
        # recalls.append(recall)
        # precisions.append(precision)
        # recall1, precision1 = eval_keyframe(label_file, video_path, 0.1, smooth=False)
        # recalls_wo_smooth.append(recall1)
        # precisions_wo_smooth.append(precision1)
    # plot_recall_precision()



    # key_frame_extract('../../data/seg_sample')
    # video_file = '../../data/sample_test/pull ups-sample1.mp4'
    # save_path = "../../plots/exp_contrast/"
    # save_path1 = save_path + os.path.basename(video_file).split('.')[0] + '_1.png'
    # save_path2 = save_path + os.path.basename(video_file).split('.')[0] + '_2.png'
    # hand_path = "D:\\Desktop\\seg_sample\\seg3-keyframe"
    # cat_keyframe = None
    # for root, _, files in os.walk(hand_path):
    #     for i, file in enumerate(files):
    #         img = cv2.imread(os.path.join(root, file))
    #         img = cv2.resize(img, (512, 910))
    #         if i == 0:
    #             cat_keyframe = img.copy()
    #         else:
    #             cat_keyframe = np.concatenate((cat_keyframe, img), axis=1)
    # video_array = gen_video_array(video_file)
    # mov_pix_vec = get_diff_pix_vec(video_array, interval=1, DEBUG=False)
    # # plotExponentialSmoothing(mov_pix_vec, [0.5, 0.1], save_path1)
    # name = os.path.basename(video_file).split('.')[0] + '.png'
    # save_path = os.path.split(os.path.realpath(__file__))[0] + '/../../plots/exp_contrast/' + name
    # mov_pix_vec = double_exponential_smoothing(mov_pix_vec, alpha=0.3, beta=0.3, save_path=save_path)
    # plotDoubleExponentialSmoothing(mov_pix_vec, alphas=[0.5, 0.3], betas=[0.9, 0.3], save_path=save_path2)
    # idxs = get_static_frames_idx(video_array, DEBUG=False, video_name=video_file)
    # print(idxs)
    # for i, idx in enumerate(idxs):
    #     if i == 0:
    #         cat_keyframe = video_array[idx]
    #     else:
    #         cat_keyframe = np.concatenate((cat_keyframe, video_array[idx]), axis=1)
    # cv2.imwrite(video_file.split('.')[0]+'_hand.png', cat_keyframe)
    print('Processing time is %.6f s' % (time.time() - begin))

