import cv2
import os
import time
import numpy as np


def gen_video_array(video_file):
    video = cv2.VideoCapture(video_file)
    frame_list = []
    # 遍历视频的每一帧
    while video.isOpened():
        # 读取下一帧
        (ret, frame) = video.read()
        frame_list.append(frame)
    frame_arr = np.array(frame_list)
    return frame_arr


if __name__ == '__main__':
    begin = time.time()
    video_file = "../../data/3.mp4"
    gen_video_array(video_file)
    print('Video reading time for 926 frames is %.6f s' % (time.time() - begin))

