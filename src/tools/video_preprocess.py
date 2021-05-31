import cv2
import os
import time
import numpy as np


def read_video(video_file, short_edge=512):
    global num_frames
    video = cv2.VideoCapture(video_file)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    src_H = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    src_W = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    if src_H > src_W:
        dst_W = int(short_edge)
        dst_H = int(short_edge * src_H // src_W)

    else:
        dst_W = int(short_edge * src_W // src_H)
        dst_H = int(short_edge)
    cnt = 0
    # 遍历视频的每一帧
    while video.isOpened():
        # 读取下一帧
        (ret, frame) = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (dst_W, dst_H))
        cnt += 1


def gen_video_list_array(video_file, short_edge=512):
    global num_frames
    video = cv2.VideoCapture(video_file)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    src_H = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    src_W = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    if src_H > src_W:
        dst_W = int(short_edge)
        dst_H = int(short_edge * src_H // src_W)

    else:
        dst_W = int(short_edge * src_W // src_H)
        dst_H = int(short_edge)
    frames = []
    cnt = 0
    # 遍历视频的每一帧
    while video.isOpened():
        # 读取下一帧
        (ret, frame) = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (dst_W, dst_H))
        frames.append(frame)
        cnt += 1
    frames = np.array(frames, dtype='uint8')
    return frames


def gen_video_array(video_file, short_edge=512):
    global num_frames
    video = cv2.VideoCapture(video_file)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    src_H = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    src_W = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    if src_H > src_W:
        dst_W = min(int(short_edge), src_W)
        dst_H = int(dst_W * src_H // src_W)

    else:
        dst_H = min(int(short_edge), src_H)
        dst_W = int(dst_H * src_W // src_H)

    frames = np.zeros([int(num_frames), int(dst_H), int(dst_W), 3], dtype='uint8')
    cnt = 0
    # 遍历视频的每一帧
    while video.isOpened():
        # 读取下一帧
        (ret, frame) = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (dst_W, dst_H))
        frames[cnt, :, :, :] = frame
        cnt += 1

    return frames


if __name__ == '__main__':
    begin = time.time()
    num_frames = 0
    video_file = "../../data/sample/front-squat-sample1.mp4"
    gen_video_array(video_file, short_edge=512)
    # read_video(video_file, short_edge=512)
    # read_video(video_file, short_edge=512)
    # read_video(video_file, short_edge=512)
    print('Video reading time for %d frames is %.6f s' % (num_frames, time.time() - begin))

