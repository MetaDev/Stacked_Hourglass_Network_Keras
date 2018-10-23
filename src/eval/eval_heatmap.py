
from eval.heatmap_process import post_process_heatmap
import numpy as np
def cal_kp_distance(pre_kp, gt_kp, threshold):
    if gt_kp[0] > 1 and gt_kp[1] > 1 :
        print(gt_kp[0:2])
        dif = np.linalg.norm(gt_kp[0:2]- pre_kp[0:2])/np.sum(gt_kp[0:2])
        if dif < threshold:
             # good prediction
            return 1
        else: # failed
            return 0
    else:
        return -1

def heatmap_accuracy(predhmap, meta, threshold):

    pred_kps  = post_process_heatmap(predhmap)
    pred_kps  = np.array(pred_kps)

    gt_kps = meta['tpts']

    good_pred_count = 0
    failed_pred_count = 0
    #TODO calcualte length of the whole skeleton
    total_kp_dist=np.sum()
    for i in range(gt_kps.shape[0]):
        dis = cal_kp_distance(pred_kps[i, :], gt_kps[i, :],  threshold)
        if dis == 0:
            failed_pred_count += 1
        elif dis  == 1:
            good_pred_count += 1

    return good_pred_count, failed_pred_count

def cal_heatmap_acc(prehmap,  metainfo, threshold):
    sum_good, sum_fail = 0,  0
    for i in range(prehmap.shape[0]):
        _prehmap = prehmap[i, :, :, :]
        good, bad = heatmap_accuracy(_prehmap, metainfo[i],  threshold=threshold)

        sum_good += good
        sum_fail += bad

    return sum_good, sum_fail