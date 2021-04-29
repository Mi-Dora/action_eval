import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt


def get_static_frames_idx(video_file):
    mov_pix_vec = get_diff_pix_vec(video_file, interval=1)
    is_static = True
    static_frames_idx = []
    thresh = mov_pix_vec.max() * 0.2
    begin = -1
    for i, n in enumerate(mov_pix_vec):
        if n < thresh and is_static is False:
            begin = i
            is_static = True
        elif n > thresh and is_static is True:
            is_static = False
            end = i
            if begin != -1:
                sliced = mov_pix_vec[begin:end]
                arg = np.argsort(sliced)
                static_frames_idx.append(arg[0]+begin)
                begin = -1
    name = os.path.basename(video_file).split('.')[0] + '.png'
    plot_move_pix(mov_pix_vec, static_frames_idx, '../../plots/' + name)
    return static_frames_idx


def get_diff_pix_vec(video_file, interval=1):
    video = cv2.VideoCapture(video_file)

    # 初始化当前帧的前两帧
    last_frame1 = None
    last_frame2 = None
    frameDelta1 = None
    move_pix_list = []
    cnt = 0
    # 遍历视频的每一帧
    while video.isOpened():
        # 读取下一帧
        (ret, frame) = video.read()
        cnt += 1
        if cnt % interval != 0:
            continue
        # 如果不能抓取到一帧，说明我们到了视频的结尾
        if not ret:
            break

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

        # 图像二值化
        thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
        move_pix_list.append(thresh.sum() / 255)

    return np.float32(move_pix_list)


def plot_move_pix(mov_pix_vec, static_frames_idx, save_path='tmp.png'):
    plt.title('Moving-Pixel Sequence')
    plt.bar(range(len(mov_pix_vec)), mov_pix_vec)
    for idx in static_frames_idx:
        plt.vlines(idx, 0, mov_pix_vec.max(), colors="r", linestyles="dashed")
    plt.xlabel('Frame ID')
    plt.ylabel('Number of moving pixel')
    plt.savefig(save_path)
    plt.clf()


if __name__ == '__main__':
    begin = time.time()
    video_file = "../../data/3.mp4"
    get_static_frames_idx(video_file)
    print('Video reading time for 926 frames is %.6f s' % (time.time() - begin))

