import cv2
import os
import sys
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import tqdm
from src.tools.video_preprocess import gen_video_array
from src.tools.frame_differential import get_static_frames_idx


def compute_eval_key_frame(extracted, gt, thresh=0.1):
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


def plot_recall_precision():
    plt.figure(figsize=(6, 4))
    # plt.ylim([0, 1.1])
    plt.plot(threshes, recalls_final, marker='*', label='Average Recall')
    plt.plot(threshes, recalls, marker='^', label='w/o auto-modify')
    plt.plot(threshes, recalls_wo_smooth, marker='o', ls="dashed", label='w/o (auto-modify + smooth)')
    # plt.title('Period-Similarity Bar')
    plt.legend()
    plt.xlabel('Thresh')
    plt.ylabel('Average Recall')
    plt.grid()
    plt.savefig('../plots/eval/recall.png')
    plt.show()
    plt.clf()

    plt.figure(figsize=(6, 4))
    # plt.ylim([0, 1.1])
    plt.plot(threshes, precisions_final, marker='*', label='Average Precision')
    plt.plot(threshes, precisions, marker='^', label='w/o auto-modify')
    plt.plot(threshes, precisions_wo_smooth, marker='o', ls="dashed", label='w/o (auto-modify + smooth)')
    # plt.title('Period-Similarity Bar')
    plt.legend()
    plt.xlabel('Thresh')
    plt.ylabel('Average Precision')
    plt.grid()
    plt.savefig('../plots/eval/precision.png')
    plt.show()
    plt.clf()


def plot_seg_eval():
    plt.figure()
    # plt.ylim([0, 1.1])
    plt.plot(IOUs, ARs, marker='*', label='AR')

    plt.plot(IOUs, AR1s, marker='^', label='AR w/o auto-modify')

    plt.plot(IOUs, AR2s, marker='o', label='AR w/o (auto-modify+smooth)')
    # plt.title('Period-Similarity Bar')
    plt.legend()
    plt.xlabel('IOU')
    plt.ylabel('Average Recall')
    plt.grid()
    plt.savefig('../plots/eval/segment_AR.png')
    plt.show()
    plt.clf()

    plt.figure()
    plt.plot(IOUs, APs, marker='*', label='AP')
    plt.plot(IOUs, AP1s, marker='^', label='AP w/o auto-modify')
    plt.plot(IOUs, AP2s, marker='o', label='AP w/o (auto-modify+smooth)')
    plt.legend()
    plt.xlabel('IOU')
    plt.ylabel('Average Precision')
    plt.grid()
    plt.savefig('../plots/eval/segment_AP.png')
    plt.show()
    plt.clf()





def load_json(file):
    f = open(file, 'r')
    content = f.read()
    contente_dict = json.loads(content)
    return contente_dict


def get_seg(static_idx, labels):
    seg_label = -1
    mid_label = -1
    for i in range(len(static_idx)):
        if seg_label == -1:
            seg_label = labels[i]
            continue
        if labels[i] != seg_label:
            mid_label = labels[i]
            break
    start = -1
    mid = -1
    segments = []
    for i in range(len(static_idx)):
        if labels[i] == seg_label and start == -1:
            start = static_idx[i]
            continue
        if labels[i] == mid_label and start != -1 and mid == -1:
            mid = static_idx[i]
            continue
        if labels[i] == seg_label and mid != -1:
            end = static_idx[i]
            segments.append([start, mid, end])
            start = end
            mid = -1
    return segments


def eval_keyframe(extracted_dict, gt_dict, thresh=0.3):
    print("Threshold = {}".format(thresh))
    tt_gt = 0
    tt_ex = 0
    tt_tp = 0
    for name in gt_dict.keys():
        extracted = np.array(extracted_dict[name])
        gt = np.array(gt_dict[name])
        TP, recall, precision = compute_eval_key_frame(extracted, gt, thresh)
        print("{}: recall={}, precision={}".format(name, recall, precision))
        tt_gt += gt.size
        tt_ex += extracted.size
        tt_tp += TP
    AR = tt_tp / tt_gt
    AP = tt_tp / tt_ex
    print("AR={}, AP={}".format(AR, AP))
    return AR, AP


def intersect(s1, e1, s2, e2):
    if s2 < e1 and s1 < e2:
        return True
    else:
        return False


def eval_seg(seg_dict, gt_dict, IOU=0.5):
    tt_gt = 0
    tt_op = 0
    tt_tp = 0
    for name in gt_dict.keys():
        output_seg = get_seg(seg_dict[name]['idx'], seg_dict[name]['label'])
        gt_seg = get_seg(gt_dict[name]['idx'], gt_dict[name]['label'])
        gt_match = np.zeros(len(gt_seg)*2)
        op_match = np.zeros(len(output_seg)*2)
        for o_seg in output_seg:
            for i, g_seg in enumerate(gt_seg):
                if intersect(o_seg[0], o_seg[1], g_seg[0], g_seg[1]):
                    intersection = min(o_seg[1], g_seg[1]) - max(o_seg[0], g_seg[0])
                    union = max(o_seg[1], g_seg[1]) - min(o_seg[0], g_seg[0])
                    cur_IOU = intersection / union
                    if cur_IOU > gt_match[i*2]:
                        gt_match[i*2] = cur_IOU
                if intersect(o_seg[1], o_seg[2], g_seg[1], g_seg[2]):
                    intersection = min(o_seg[2], g_seg[2]) - max(o_seg[1], g_seg[1])
                    union = max(o_seg[2], g_seg[2]) - min(o_seg[1], g_seg[1])
                    cur_IOU = intersection / union
                    if cur_IOU > gt_match[i*2+1]:
                        gt_match[i*2+1] = cur_IOU

        TP = gt_match >= IOU
        tt_tp += TP.sum()
        tt_gt += len(gt_seg*2)
        tt_op += len(output_seg*2)
    AR = tt_tp / tt_gt
    AP = tt_tp / tt_op
    print("IOU={}: AR={}, AP={}".format(IOU, AR, AP))
    return AR, AP


def eval_seg2(seg_dict, gt_dict, IOU=0.5):
    tt_gt = 0
    tt_op = 0
    tt_tp = 0
    for name in gt_dict.keys():
        output_seg = get_seg(seg_dict[name]['idx'], seg_dict[name]['label'])
        gt_seg = get_seg(gt_dict[name]['idx'], gt_dict[name]['label'])
        gt_match = np.zeros(len(gt_seg))
        op_match = np.zeros(len(output_seg))
        for o_seg in output_seg:
            for i, g_seg in enumerate(gt_seg):
                if intersect(o_seg[0], o_seg[2], g_seg[0], g_seg[2]):
                    intersection = min(o_seg[2], g_seg[2]) - max(o_seg[0], g_seg[0])
                    union = max(o_seg[2], g_seg[2]) - min(o_seg[0], g_seg[0])
                    cur_IOU = intersection / union
                    if cur_IOU > gt_match[i]:
                        gt_match[i] = cur_IOU

        TP = gt_match >= IOU
        tt_tp += TP.sum()
        tt_gt += len(gt_seg)
        tt_op += len(output_seg)
    AR = tt_tp / tt_gt
    AP = tt_tp / tt_op
    print("IOU={}: AR={}, AP={}".format(IOU, AR, AP))
    return AR, AP


if __name__ == '__main__':
    begin = time.time()
    seg_label_file = "../eval/seg_label.json"
    seg_no_smooth = "../eval/seg_nothing.json"
    seg_no_modify = "../eval/seg_no_modify.json"
    seg_file = "../eval/final_seg.json"
    key_file = "../eval/keyframe.json"
    key_unsmooth_file = "../eval/keyframe_unsmooth.json"
    key_label_file = "../eval/keyframe_label.json"
    seg_gt = load_json(seg_label_file)
    seg = load_json(seg_file)
    seg_wo_s = load_json(seg_no_smooth)
    seg_wo_m = load_json(seg_no_modify)
    key = load_json(key_file)
    key_unsmooth = load_json(key_unsmooth_file)
    key_gt = load_json(key_label_file)

    # # Key Frame Eval
    # key_final = {}
    # for name in seg.keys():
    #     key_final[name] = seg[name]['idx']
    # threshes = np.arange(0.02, 0.25, 0.02)
    # recalls = []
    # recalls_wo_smooth = []
    # recalls_final = []
    # precisions = []
    # precisions_wo_smooth = []
    # precisions_final = []
    #
    # for thresh in threshes:
    #     recall, precision = eval_keyframe(key, key_gt, thresh)
    #     recalls.append(recall)
    #     precisions.append(precision)
    #     recall1, precision1 = eval_keyframe(key_unsmooth, key_gt, thresh)
    #     recalls_wo_smooth.append(recall1)
    #     precisions_wo_smooth.append(precision1)
    #     recall2, precision2 = eval_keyframe(key_final, key_gt, thresh)
    #     recalls_final.append(recall2)
    #     precisions_final.append(precision2)
    # plot_recall_precision()

    # Segment Eval
    IOUs = np.arange(0.5, 1.0, 0.025)
    ARs = []
    APs = []
    AR1s = []
    AP1s = []
    AR2s = []
    AP2s = []
    for IOU in IOUs:
        print('Final:')
        recall, precision = eval_seg(seg, seg_gt, IOU)
        ARs.append(recall)
        APs.append(precision)
        print('\n')
        recall1, precision1 = eval_seg(seg_wo_m, seg_gt, IOU)
        AR1s.append(recall1)
        AP1s.append(precision1)
        print('\n')
        recall2, precision2 = eval_seg(seg_wo_s, seg_gt, IOU)
        AR2s.append(recall2)
        AP2s.append(precision2)

    plot_seg_eval()
    print('Processing time is %.6f s' % (time.time() - begin))







