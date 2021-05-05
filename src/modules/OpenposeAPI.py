#!/usr/bin/env python
import os
import sys
from sys import platform
import time
import numpy as np
import cv2
import tqdm
from src.tools.video_preprocess import gen_video_array


openpose_header = [
    'nose', 'neck',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'right_eye', 'left_eye',
    'right_ear', 'left_ear'
]


class OpenPoseEstimator(object):
    def __init__(self, op_path, DEBUG=False):
        self.op_path = op_path
        self.DEBUG = DEBUG
        self.H = -1
        self.W = -1
        # load openpose python api
        if self.op_path is not None:
            try:
                # Windows Import
                if platform == "win32":
                    # Change these variables to point to the correct folder (Release/x64 etc.)
                    sys.path.append(self.op_path + '/build/python/openpose/Release')
                    os.environ['PATH'] = os.environ['PATH'] + ';' + self.op_path + '/build/x64/Release;' + self.op_path + '/build/bin;'
                    import pyopenpose as op
                else:
                    # Change these variables to point to the correct folder (Release/x64 etc.)
                    sys.path.append('../../python')
                    # If you run `make install` (default path is `/usr/local/python` for Ubuntu),
                    # you can also access the OpenPose/python module from there.
                    # This will install OpenPose and the python library at your desired installation path.
                    # Ensure that this is in your python path in order to use it.
                    # sys.path.append('/usr/local/python')
                    from OpenposeAPI import pyopenpose as op
            except ImportError:
                print('Can not find Openpose Python API.')
                return

        # initiate
        self.op = op
        self.opWrapper = op.WrapperPython()
        params = dict(model_folder='D:\\Desktop\\Grad_poj\\Action-Evaluation\\weights', model_pose='COCO')
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def estimate_video(self, video_array, interval=1, only_prior_person=True):
        num_frames = video_array.shape[0]
        pbar = tqdm.tqdm(total=num_frames)
        # start recognition
        pose_list = []
        norm_pose_list = []
        output_list = []
        for i in range(num_frames):
            tic = time.time()
            if i % interval != 0:
                continue
            # get image
            frame = video_array[i]

            # pose estimation
            datum = self.op.Datum()
            datum.cvInputData = frame
            self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
            multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)

            if only_prior_person:
                pose = self.__get_prior_person(multi_pose)
                pose_list.append(pose)
                norm_pose_list.append(self.__pose_normalization(pose))
            else:
                pose_list.append(multi_pose)
                norm_pose_list.append(self.__pose_normalization(multi_pose))
            if len(multi_pose.shape) != 3:
                continue
            output = datum.cvOutputData.copy()
            output_list.append(output)
            if self.DEBUG:
                fps = 1 / (time.time() - tic)
                cv2.putText(output, '%.2f fps' % fps, (30, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0),
                            thickness=2)
                cv2.imshow("Output", output)
                cv2.waitKey(1)
            pbar.update(interval)
        pbar.close()
        return pose_list, norm_pose_list, output_list

    # def estimate_video(self, video_file, short_edge=512, only_prior_person=True):
    #     input_movie = cv2.VideoCapture(video_file)
    #     num_frames = input_movie.get(cv2.CAP_PROP_FRAME_COUNT)
    #     pbar = tqdm.tqdm(total=num_frames)
    #     # start recognition
    #     start_time = time.time()
    #     frame_index = 0
    #     pose_list = []
    #     norm_pose_list = []
    #     output_list = []
    #     while True:
    #         tic = time.time()
    #         # get image
    #         ret, frame = input_movie.read()
    #         if not ret:
    #             break
    #         source_H, source_W, _ = frame.shape
    #         if source_H > source_W:
    #             frame = cv2.resize(
    #                 frame, (short_edge, short_edge * source_H // source_W))
    #         else:
    #             frame = cv2.resize(
    #                 frame, (short_edge * source_W // source_H, short_edge))
    #         self.H, self.W, _ = frame.shape
    #
    #         # pose estimation
    #         datum = self.op.Datum()
    #         datum.cvInputData = frame
    #         self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
    #         multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)
    #
    #         if only_prior_person:
    #             pose = self.__get_prior_person(multi_pose)
    #             pose_list.append(pose)
    #             norm_pose_list.append(self.__pose_normalization(pose))
    #         else:
    #             pose_list.append(multi_pose)
    #             norm_pose_list.append(self.__pose_normalization(multi_pose))
    #         if len(multi_pose.shape) != 3:
    #             continue
    #         output = datum.cvOutputData.copy()
    #         output_list.append(output)
    #         if self.DEBUG:
    #             fps = 1 / (time.time() - tic)
    #             cv2.putText(output, '%.2f fps' % fps, (30, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0),
    #                         thickness=2)
    #             cv2.imshow("Output", output)
    #             cv2.waitKey(1)
    #         pbar.update(1 )
    #     pbar.close()
    #     return pose_list, norm_pose_list, output_list

    def estimate_image(self, img, short_edge=512, only_prior_person=True):
        source_H, source_W, _ = img.shape
        if source_H > source_W:
            img = cv2.resize(
                img, (short_edge, short_edge * source_H // source_W))
            self.H, self.W, _ = img.shape
        else:
            img = cv2.resize(
                img, (short_edge * source_W // source_H, short_edge))
        # pose estimation
        datum = self.op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)
        output_img = datum.cvOutputData.copy()
        if only_prior_person:
            pose = self.__get_prior_person(multi_pose)
            return pose, self.__pose_normalization(pose), output_img
        return multi_pose, self.__pose_normalization(multi_pose), output_img

    def __get_prior_person(self, multi_pose):
        max_conf = -1
        prior = -1
        num_people, _, _ = multi_pose.shape
        for person in range(num_people):
            conf = multi_pose[person, :, 2].sum()
            if conf > max_conf:
                prior = person
        if prior != -1:
            return multi_pose[np.newaxis, prior, :, :]
        else:
            print('No person detected!')
            return None

    def __pose_normalization(self, pose):
        # pose shape: (num_person, num_joint, 3)
        norm_pose = pose.copy()
        norm_pose[:, :, 0] = pose[:, :, 0] / self.W
        norm_pose[:, :, 1] = pose[:, :, 1] / self.H
        norm_pose[:, :, 0:2] = norm_pose[:, :, 0:2] - 0.5
        norm_pose[:, :, 0][pose[:, :, 2] == 0] = 0
        norm_pose[:, :, 1][pose[:, :, 2] == 0] = 0
        return norm_pose


if __name__ == '__main__':
    openpose = OpenPoseEstimator('D:\\Desktop\\openpose', DEBUG=True)
    sample_video = "../../data/3.mp4"
    short_edge = 512

    video_array = gen_video_array(sample_video, short_edge)
    openpose.estimate_video(video_array)
