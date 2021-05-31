import numpy as np
import cv2
import os
from src.tools.frame_differential import get_static_frames_idx
from src.tools.pose_angle import get_video_angle_vec
from src.tools.transforms import get_transform_matrix, transform_pose
from src.tools.utils import plot_skeleton, array2dict, save_video, pose_patch
from src.tools.eval_metrics import eval, cal_warping_path, video_warping, plot_path, pose_warping1
from src.modules.OpenposeAPI import OpenPoseEstimator

from src.tools.video_preprocess import gen_video_array
from src.tools.video_segmentation import get_seg_interval


def video_eval(openpose_estimator, sample_video, template_video=None, short_edge=512, interval=1):
    print('Starting computing static frames...')
    video_array = gen_video_array(sample_video, short_edge)
    # video_array = video_array[0:100]
    static_frames_idx = get_static_frames_idx(video_array, video_name=sample_video, interval=interval)

    print('Starting estimating poses...')
    npy_path = sample_video[:-4] + '.npy'
    norm_npy_path = sample_video[:-4] + '_norm.npy'
    if os.path.exists(npy_path) and os.path.exists(norm_npy_path):
        print(npy_path + ' and ' + norm_npy_path + 'found.')
        src_pose_arr = np.load(npy_path)
        norm_src_pose_arr = np.load(norm_npy_path)
    else:
        src_pose_list, norm_src_pose_list, output_list = openpose_estimator.estimate_video(video_array, interval=interval)
        src_pose_arr = np.squeeze(np.float64(src_pose_list))
        norm_src_pose_arr = np.float64(norm_src_pose_list)
        np.save(npy_path, src_pose_arr)
        print(npy_path + ' and ' + norm_npy_path + 'saved.')

    print('Starting patching openpose estimation...')
    src_pose_arr = pose_patch(src_pose_arr)
    for pose, frame in zip(src_pose_arr, video_array):
        pose = array2dict(pose)
        plot_skeleton(frame, pose, color=(0, 255, 0))
        # cv2.imshow('Plot', frame)
        # cv2.waitKey(30)
    # save_video(video_array, '../video/before_patch.mp4')

    print('Starting segmenting video...')
    angle_vec_arr = get_video_angle_vec(src_pose_arr)
    segments = get_seg_interval(static_frames_idx, angle_vec_arr)
    template_angle_vec = angle_vec_arr[segments[0][0]:segments[0][2]]
    template_pose_arr = src_pose_arr[segments[0][0]:segments[0][2]]
    template_video_array = video_array[segments[0][0]:segments[0][2]]
    del segments[0]
    cat_videos = []
    seg_video_scores = []
    for i in range(0, len(segments)):
        print('Processing Segment ' + str(i+1))
        dis, path = cal_warping_path(angle_vec_arr[segments[i][0]:segments[i][2]], template_angle_vec)
        # frame in list
        cat_video = video_warping(video_array[segments[i][0]:segments[i][2]], template_video_array, path)
        input_pose_list, temp_pose_list = pose_warping1(src_pose_arr[segments[i][0]:segments[i][2]], template_pose_arr, path)
        # plot_path(path, '../plots/DTW_path_{}_{}.png'.format(os.path.basename(sample_video).split('.')[0], i))
        scores = []
        for j in range(len(cat_video)):
            temp_pose = array2dict(temp_pose_list[j])
            input_pose = array2dict(input_pose_list[j])
            trans_mat, _ = get_transform_matrix(temp_pose, input_pose, method='SoftRigid')
            dst_pose = transform_pose(trans_mat, temp_pose)
            score = eval(dst_pose, input_pose, facial_feature=False, method='angle')
            scores.append(score)
            # Since input video is on the left of the concatenated video,
            # so the pose coordinate do not need to translate
            plot_skeleton(cat_video[j], dst_pose, color=(255, 0, 0))
            canvas = np.zeros_like(video_array[j])
            cv2.putText(cat_video[j], 'Sample', (10, 150),
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 0, 0), thickness=2)
            cv2.putText(cat_video[j], 'Template', (10 + cat_video[j].shape[1]//2, 150),
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 0, 0), thickness=2)
            cv2.putText(canvas, 'Count: %d' % i, (10, 150),
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), thickness=2)
            cv2.putText(canvas, 'Score for current frame: %.4f' % score, (10, 230),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            for k, seg_video_score in enumerate(seg_video_scores):
                cv2.putText(canvas, 'Score for period %d: %.4f' % (k+1, seg_video_score), (10, 230 + (k+1)*50),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            cat_videos.append(np.concatenate((canvas, cat_video[j]), axis=1))
        scores = np.float64(scores)
        seg_video_scores.append(scores.sum() / scores.size)
    return cat_videos


if __name__ == '__main__':
    # sample_video = "../data/3.mp4"
    data_path = '../test_data'
    sample_videos = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.mp4'):
                sample_videos.append(os.path.join(root, file))
    # template_video = ""
    op = OpenPoseEstimator('D:\\Desktop\\openpose', DEBUG=False)
    for sample_video in sample_videos:
        print(sample_video)
        export_video = video_eval(op, sample_video, template_video=None, short_edge=512, interval=1)
        save_video(export_video, '../video/' + os.path.basename(sample_video))


