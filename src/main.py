import numpy as np
import cv2
from src.tools.frame_differential import get_static_frames_idx
from src.tools.pose_angle import get_video_angle_vec
from src.tools.transforms import get_transform_matrix, transform_pose
from src.tools.utils import plot_skeleton, array2dict
from src.tools.eval_metrics import eval, sequence_warping, video_warping, plot_path
from src.modules.OpenposeAPI import OpenPoseEstimator
from src.tools.video_preprocess import gen_video_array
from src.tools.video_segmentation import get_seg_interval


sample_video = "../data/3.mp4"
reference_img = cv2.imread('../sample/test.png')
short_edge = 512
interval = 1

video_array = gen_video_array(sample_video, short_edge)
static_frames_idx = get_static_frames_idx(video_array, video_name=sample_video, interval=interval)

op = OpenPoseEstimator('D:\\Desktop\\openpose', DEBUG=False)

ref_pose, norm_ref_pose, ref_output = op.estimate_image(reference_img, short_edge=short_edge)

src_pose_list, norm_src_pose_list, output_list = op.estimate_video(video_array, interval=interval)

ref_pose = array2dict(ref_pose)
angle_vec_arr = get_video_angle_vec(src_pose_list)
segments = get_seg_interval(static_frames_idx, angle_vec_arr)
src_pose_arr = np.float64(src_pose_list)
print(segments[0][2]-segments[0][0])
print(segments[1][2]-segments[1][0])
dis, path = sequence_warping(angle_vec_arr[segments[0][0]:segments[0][2]], angle_vec_arr[segments[1][0]:segments[1][2]])
new_video = video_warping(video_array[segments[0][0]:segments[0][2]], video_array[segments[1][0]:segments[1][2]], path)
plot_path(path, '../plots/DTW_path_3')
for frame in new_video:
    cv2.imshow('Warping', frame)
    cv2.waitKey(100)

# for src_pose, src_output in zip(src_pose_list, output_list):
#     src_pose = array2dict(src_pose)
#     mat1, mat2 = get_transform_matrix(src_pose, ref_pose, method='SoftRigid')
#     dst_pose = transform_pose(mat1, src_pose)
#     dst_pose2 = transform_pose(mat2, src_pose)
#     score = eval(dst_pose, ref_pose, facial_feature=False, method='angle')
#
#     plot_skeleton(ref_output, dst_pose, color=(0, 255, 0))
#
#     cv2.putText(ref_output, 'Score: %.4f' % score, (30, 330), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), thickness=2)
#     img = np.concatenate((src_output, ref_output), axis=1)
#     cv2.imshow('../test8.png', img)
#     # cv2.imshow('src', src_output)
#     # cv2.imshow('ref', ref_output)
#     cv2.waitKey(0)
#     print('Done.')






