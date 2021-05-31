import numpy as np
import cv2
import os
import time
from src.tools.frame_differential import get_static_frames_idx
from src.tools.pose_angle import get_video_angle_vec, plot_angle_curve_seg, plot_angle_curve
from src.tools.transforms import get_transform_matrix, transform_pose
from src.tools.utils import plot_skeleton, array2dict, save_video, pose_patch, plot_score_curve, plot_bar, pose_smooth
from src.tools.utils import init_video_writer, plot_pose_traj
from src.tools.eval_metrics import eval, cal_warping_path, video_warping, pose_warping1, pose_warping2, cal_seg_score
from src.tools.eval_metrics import plot_path, get_pose_angle_mask
from src.modules.stgcn.stgcnAPI_2 import StGcn
from src.modules.OpenposeAPI import OpenPoseEstimator
from src.tools.video_preprocess import gen_video_array
from src.tools.video_segmentation import get_seg_interval
from src.tools.pca import PCA


def sample_video_process(src_pose_arr, video_array, static_frames_idx, action_name):
    for pose, frame in zip(src_pose_arr, video_array):
        pose = array2dict(pose)
        plot_skeleton(frame, pose, color=(0, 255, 0), thick=3*coef)
    begin = time.time()
    print('Starting segmenting video...')
    angle_vec_arr, mask = get_video_angle_vec(src_pose_arr, action_class=action_name)
    # get_static_frames_idx_pca(angle_vec_arr, mask.sum())
    segments, dispersion = get_seg_interval(static_frames_idx, angle_vec_arr, video_name=os.path.basename(sample_video))
    template_pose_arr, template_angle_vec, template_video_array, mid_frame = template_process(
        action_name, template_path, op_path, short_edge, interval)
    # plot_angle_curve(template_angle_vec, name='original')
    # pca_estimator = PCA(percentage=0.95)
    # mask = pca_estimator.fit_transform(angle_vec_arr)
    # print('Mask: {}'.format(mask))
    # plot_angle_curve(template_angle_vec, name='pca')
    # copied = angle_vec_arr.copy()
    # angle_vec_arr = pca_estimator.transform(angle_vec_arr)

    cat_videos = []
    last_frame = None
    seg_video_scores = []
    score_list = []
    paths = []
    for i in range(0, len(segments)):
        # print('Processing Segment ' + str(i + 1))
        # print(action_name)
        dis1, path1 = cal_warping_path(angle_vec_arr[segments[i][0]:segments[i][1]], template_angle_vec[:mid_frame],
                                       mask)
        dis2, path2 = cal_warping_path(angle_vec_arr[segments[i][1]:segments[i][2]], template_angle_vec[mid_frame:],
                                       mask)

        path1, path1_scores = pose_warping2(src_pose_arr[segments[i][0]:segments[i][1]],
                                            template_pose_arr[:mid_frame], path1, mask)
        path2, path2_scores = pose_warping2(src_pose_arr[segments[i][1]:segments[i][2]],
                                            template_pose_arr[mid_frame:], path2, mask)
        paths.append([path1, path2])
        input_pose_list1, temp_pose_list1 = pose_warping1(src_pose_arr[segments[i][0]:segments[i][1]],
                                                          template_pose_arr[:mid_frame], path1)
        input_pose_list2, temp_pose_list2 = pose_warping1(src_pose_arr[segments[i][1]:segments[i][2]],
                                                          template_pose_arr[mid_frame:], path2)
        cat_video12 = [video_array[segments[i][0]:segments[i][1]], video_array[segments[i][1]:segments[i][2]]]
        input_pose_list12 = [input_pose_list1, input_pose_list2]
        temp_pose_list12 = [temp_pose_list1, temp_pose_list2]
        part = 0
        seg_score = []
        for cat_video, input_pose_list, temp_pose_list in zip(cat_video12, input_pose_list12, temp_pose_list12):
            scores = []
            for j in range(len(cat_video)):
                temp_pose = array2dict(temp_pose_list[j])
                input_pose = array2dict(input_pose_list[j])
                trans_mat, _ = get_transform_matrix(temp_pose, input_pose, method='SoftRigid')
                dst_pose = transform_pose(trans_mat, temp_pose)
                score = eval(dst_pose, input_pose, facial_feature=False, method='angle', orient_weight=1.0, mask=mask)
                scores.append(score)
                score_list.append(score)
                # Since input video is on the left of the concatenated video,
                # so the pose coordinate do not need to translate
                plot_skeleton(cat_video[j], dst_pose, color=(255, 0, 0), thick=3*coef)
                canvas = np.zeros_like(video_array[j])
                cv2.putText(cat_video[j], 'Sample', (10*coef, 150*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (255, 0, 0), thickness=2*coef)
                cv2.putText(canvas, 'Class: %s' % action_name, (10*coef, 50*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (0, 255, 0), thickness=2*coef)
                cv2.putText(canvas, 'Dispersion: %.4f' % dispersion, (10*coef, 100*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (0, 255, 0), thickness=2*coef)
                cv2.putText(canvas, 'Count: %d/%d' % (i + part, len(segments)), (10*coef, 150*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (0, 255, 0), thickness=2*coef)
                cv2.putText(canvas, 'Score for current frame: %.2f' % (score * 100), (10*coef, 230*coef),
                            cv2.FONT_HERSHEY_PLAIN, 1.5*coef, (0, 255, 0), thickness=2*coef)
                for k, seg_video_score in enumerate(seg_video_scores):
                    cv2.putText(canvas, 'Score for period %d: %.2f' % (k + 1, seg_video_score * 100),
                                (10*coef, (230 + (k + 1) * 50)*coef),
                                cv2.FONT_HERSHEY_PLAIN, 1.5*coef, (0, 255, 0), thickness=2*coef)
                # cat_videos.append(np.concatenate((canvas, cat_video[j]), axis=1))
                last_frame = np.concatenate((canvas, cat_video[j]), axis=1)
                video_writer.write(last_frame)
            part += 1
            scores = np.float64(scores)
            seg_score.append(cal_seg_score(scores))
        seg_video_scores.append((seg_score[0] + seg_score[1]) / 2)

    # To show the last period score
    # last_frame = cat_videos[-1].copy()
    cv2.putText(last_frame, 'Score for period %d: %.2f' % (len(seg_video_scores), seg_video_scores[-1]*100),
                (10*coef, (230 + len(seg_video_scores) * 50)*coef),
                cv2.FONT_HERSHEY_PLAIN, 1.5*coef, (0, 255, 0), thickness=2*coef)
    # cat_videos.append(last_frame)
    for _ in range(10):
        video_writer.write(last_frame)
    print('Video process time is %.6f s' % (time.time() - begin))
    plot_path(paths, '../../../plots/DTW-path/{}.png'.format(os.path.basename(sample_video).split('.')[0]))
    plot_score_curve(score_list, segments,
                     '../../../plots/similarity/{}.png'.format(os.path.basename(sample_video).split('.')[0]))
    plot_bar(seg_video_scores,
             '../../../plots/bar/{}.png'.format(os.path.basename(sample_video).split('.')[0]))
    plot_angle_curve_seg(angle_vec_arr, segments,
                         save_path='../../../plots/angle/{}.png'.format(os.path.basename(sample_video).split('.')[0]))
    return cat_videos


def template_process(action_name, template_path, op_path, short_edge, interval, plot=True):
    global orient
    template_video = None
    for root, _, files in os.walk(template_path):
        for file in files:
            if file.startswith(action_name) and file.endswith('mp4'):
                if action_name == 'squat' or action_name == 'pull ups':
                    print(file.split('-')[2])
                    print(orient)
                    if file.split('-')[2] == orient:
                        template_video = os.path.join(root, file)
                        break
                else:
                    template_video = os.path.join(root, file)
                    break
    if template_video is None:
        print('No corresponding Template Video.')
        return None
    mid_frame = int(template_video.split('-')[1])
    op = OpenPoseEstimator(op_path, DEBUG=False)
    video_array = gen_video_array(template_video, short_edge)
    npy_path = template_video[:-4] + '.npy'
    print('Starting processing template video...')
    if os.path.exists(npy_path):
        print(npy_path + ' found.')
        src_pose_arr = np.load(npy_path)
    else:
        src_pose_list, norm_src_pose_list, output_list = op.estimate_video(video_array, interval=interval)
        src_pose_arr = np.squeeze(np.float64(src_pose_list))
        np.save(npy_path, src_pose_arr)
        print(npy_path + ' saved.')
    # if EVAL:
    #     copy_pose = src_pose_arr.copy()
    src_pose_arr = pose_patch(src_pose_arr)
    src_pose_arr = pose_smooth(src_pose_arr)
    # if EVAL:
    #     plot_pose_traj(copy_pose, src_pose_arr, action_name, part='elbow')
    if plot:
        for pose, frame in zip(src_pose_arr, video_array):
            pose = array2dict(pose)
            plot_skeleton(frame, pose, color=(0, 255, 0), thick=3*coef)
    angle_vec_arr, _ = get_video_angle_vec(src_pose_arr, action_class=action_name)

    return src_pose_arr, angle_vec_arr, video_array, mid_frame


def video_eval(stgcn_predictor, sample_video, template_path=None):
    stgcn_predictor.set_param(op_path, sample_video, short_edge)
    print('Starting computing static frames...')
    video_array = gen_video_array(sample_video, short_edge)
    # video_array = video_array[0:100]
    static_frames_idx = get_static_frames_idx(video_array, video_name=sample_video, interval=interval)

    print('Starting estimating poses...')
    npy_path = sample_video[:-4] + '.npy'
    # norm_npy_path = sample_video[:-4] + '_stgcn_data_numpy.npy'
    class_path = sample_video[:-4] + '_action_class.txt'
    if os.path.exists(npy_path) and os.path.exists(class_path):
        print(npy_path + ' and ' + class_path + ' found.')
        src_pose_arr = np.load(npy_path)
        # data_numpy = np.load(norm_npy_path)
        with open(class_path, 'r') as f:
            action_name = f.readline()
    else:
        src_pose_list, action_name, data_numpy = stgcn_predictor.start()
        print('Action Class: ' + action_name)
        src_pose_arr = np.squeeze(np.float64(src_pose_list))
        with open(class_path, 'w') as f:
            f.write(action_name)
        np.save(npy_path, src_pose_arr)
        # np.save(norm_npy_path, data_numpy)
        print(npy_path + ' and ' + class_path + ' saved.')

    print('Starting patching openpose estimation...')
    src_pose_arr = pose_patch(src_pose_arr)
    src_pose_arr = pose_smooth(src_pose_arr)
    if only_sample:
        return sample_video_process(src_pose_arr, video_array, static_frames_idx, action_name)
    for pose, frame in zip(src_pose_arr, video_array):
        pose = array2dict(pose)
        plot_skeleton(frame, pose, color=(0, 255, 0), thick=3*coef)
    print('Starting segmenting video...')
    angle_vec_arr, mask = get_video_angle_vec(src_pose_arr, action_class=action_name)
    segments, dispersion = get_seg_interval(static_frames_idx, angle_vec_arr, video_name=os.path.basename(sample_video))
    template_pose_arr, template_angle_vec, template_video_array, mid_frame = template_process(
        action_name, template_path, op_path, short_edge, interval)
    # get_pose_angle_mask(template_pose_arr, percent=0.9)
    last_frame = None
    cat_videos = []
    seg_video_scores = []
    for i in range(0, len(segments)):
        print('Processing Segment ' + str(i + 1))
        # print(action_name)
        dis1, path1 = cal_warping_path(angle_vec_arr[segments[i][0]:segments[i][1]], template_angle_vec[:mid_frame],
                                       mask)
        dis2, path2 = cal_warping_path(angle_vec_arr[segments[i][1]:segments[i][2]], template_angle_vec[mid_frame:],
                                       mask)
        # frame in list
        cat_video1 = video_warping(video_array[segments[i][0]:segments[i][1]], template_video_array[:mid_frame], path1)
        cat_video2 = video_warping(video_array[segments[i][1]:segments[i][2]], template_video_array[mid_frame:], path2)
        input_pose_list1, temp_pose_list1 = pose_warping1(src_pose_arr[segments[i][0]:segments[i][1]],
                                                          template_pose_arr[:mid_frame], path1)
        input_pose_list2, temp_pose_list2 = pose_warping1(src_pose_arr[segments[i][1]:segments[i][2]],
                                                          template_pose_arr[mid_frame:], path2)
        # plot_path(path, '../plots/DTW_path_{}_{}.png'.format(os.path.basename(sample_video).split('.')[0], i))
        # scores = []
        cat_video12 = [cat_video1, cat_video2]
        input_pose_list12 = [input_pose_list1, input_pose_list2]
        temp_pose_list12 = [temp_pose_list1, temp_pose_list2]
        part = 0
        seg_score = []
        for cat_video, input_pose_list, temp_pose_list in zip(cat_video12, input_pose_list12, temp_pose_list12):
            scores = []
            for j in range(len(cat_video)):
                temp_pose = array2dict(temp_pose_list[j])
                input_pose = array2dict(input_pose_list[j])
                trans_mat, _ = get_transform_matrix(temp_pose, input_pose, method='SoftRigid')
                dst_pose = transform_pose(trans_mat, temp_pose)
                score = eval(dst_pose, input_pose, facial_feature=False, method='angle', orient_weight=1.0, mask=mask)
                scores.append(score)
                # Since input video is on the left of the concatenated video,
                # so the pose coordinate do not need to translate
                plot_skeleton(cat_video[j], dst_pose, color=(255, 0, 0), thick=3*coef)
                canvas = np.zeros_like(video_array[j])
                cv2.putText(cat_video[j], 'Sample', (10*coef, 150*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (255, 0, 0), thickness=2*coef)
                cv2.putText(cat_video[j], 'Template', (10*coef + cat_video[j].shape[1] // 2, 150*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (255, 0, 0), thickness=2*coef)
                cv2.putText(canvas, 'Class: %s' % action_name, (10*coef, 50*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (0, 255, 0), thickness=2*coef)
                cv2.putText(canvas, 'Dispersion: %.4f' % dispersion, (10*coef, 100*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (0, 255, 0), thickness=2*coef)
                cv2.putText(canvas, 'Count: %d/%d' % (i + part, len(segments)), (10*coef, 150*coef),
                            cv2.FONT_HERSHEY_PLAIN, 2.5*coef, (0, 255, 0), thickness=2*coef)
                cv2.putText(canvas, 'Score for current frame: %.2f' % (score * 100), (10*coef, 230*coef),
                            cv2.FONT_HERSHEY_PLAIN, 1.5*coef, (0, 255, 0), thickness=2*coef)
                for k, seg_video_score in enumerate(seg_video_scores):
                    cv2.putText(canvas, 'Score for period %d: %.2f' % (k + 1, seg_video_score * 100),
                                (10*coef, (230 + (k + 1) * 50)*coef),
                                cv2.FONT_HERSHEY_PLAIN, 1.5*coef, (0, 255, 0), thickness=2*coef)
                # cat_videos.append(np.concatenate((canvas, cat_video[j]), axis=1))
                last_frame = np.concatenate((canvas, cat_video[j]), axis=1)
                video_writer.write(last_frame)
            part += 1
            scores = np.float64(scores)
            seg_score.append(cal_seg_score(scores))
        seg_video_scores.append((seg_score[0] + seg_score[1]) / 2)

    # To show the last period score
    # last_frame = cat_videos[-1].copy()

    cv2.putText(last_frame, 'Score for period %d: %.2f' % (len(seg_video_scores), seg_video_scores[-1]*100),
                (10*coef, (230 + len(seg_video_scores) * 50)*coef),
                cv2.FONT_HERSHEY_PLAIN, 1.5*coef, (0, 255, 0), thickness=(2*coef))
    # cat_videos.append(last_frame)
    for _ in range(10):
        video_writer.write(last_frame)

    plot_bar(seg_video_scores,
             '../../../plots/bar/{}.png'.format(os.path.basename(sample_video).split('.')[0]))
    plot_angle_curve_seg(angle_vec_arr, segments,
                         save_path='../../../plots/angle/{}.png'.format(os.path.basename(sample_video).split('.')[0]))
    return cat_videos


if __name__ == '__main__':
    # sample_video = "../data/3.mp4"
    data_path = '../../../data/sample-512'
    template_path = '../../../data/template_test'
    sample_videos = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.mp4'):
                sample_videos.append(os.path.join(root, file))
    # # template_video = ""
    only_sample = True
    op_path = 'D:\\Desktop\\openpose'
    short_edge = 512
    interval = 1
    begin = time.time()
    stgcn = StGcn()
    stgcn.get_parser()
    for sample_video in sample_videos:
        if only_sample:
            width = 2
            save_path = '../../../video2/' + os.path.basename(sample_video)
        else:
            width = 3
            save_path = '../../../video1/' + os.path.basename(sample_video)
        print(sample_video)
        orient = os.path.basename(sample_video).split('-')[0]
        coef = short_edge // 512
        video_writer = init_video_writer(short_edge*16/9, short_edge * width, save_path)
        export_video = video_eval(stgcn, sample_video, template_path)
        video_writer.release()
        print(save_path + ' saved.')
        # if only_sample:
        #     save_video(export_video, '../../../video2/' + os.path.basename(sample_video))
        # else:
        #     save_video(export_video, '../../../video1/' + os.path.basename(sample_video))
    print('Total time consumption is %.6f s' % (time.time() - begin))
