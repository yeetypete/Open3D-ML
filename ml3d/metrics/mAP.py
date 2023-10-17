import numpy as np
from . import iou_bev, iou_3d
import torch

def filter_data(data, labels, diffs=None):
    """Filters the data to fit the given labels and difficulties.
    Args:
        data (dict): Dictionary with the data (as numpy arrays).
            {
                'label':      [...], # expected
                'difficulty': [...]  # if diffs not None
                ...
            }
        labels (number[]): List of labels which should be maintained.
        difficulties (number[]): List of difficulties which should maintained.
            (optional)

    Returns:
        Tuple with dictionary with same as format as input, with only the given labels
        and difficulties and the indices.
    """
    cond = np.any([data['label'] == label for label in labels], axis=0)
    if diffs is not None and 'difficulty' in data:
        dcond = np.any([
            np.all([data['difficulty'] >= 0, data['difficulty'] <= diff],
                   axis=0) for diff in diffs
        ],
                       axis=0)
        cond = np.all([cond, dcond], axis=0)
    idx = np.where(cond)[0]

    result = {}
    for k in data:
        result[k] = data[k][idx]
    return result, idx


def precision_3d(pred,
                 target,
                 classes=[0],
                 difficulties=[0],
                 min_overlap=[0.5],
                 bev=True,
                 similar_classes={}):
    """Computes precision quantities for each predicted box.
    Args:
        pred (dict): Dictionary with the prediction data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],
                'score':      [...],
                'difficulty': [...],
                ...
            }
        target (dict): Dictionary with the target data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],
                'difficulty': [...],
                ...
            }
        classes (number[]): List of classes which should be evaluated.
            Default is [0].
        difficulties (number[]): List of difficulties which should evaluated.
            Default is [0].
        min_overlap (number[]): Minimal overlap required to match bboxes.
            One entry for each class expected. Default is [0.5].
        bev (boolean): Use BEV IoU (else 3D IoU is used).
            Default is True.
        similar_classes (dict): Assign classes to similar classes that were not part of the training data so that they are not counted as false negatives.
            Default is {}.

    Returns:
        A tuple with a list of detection quantities
        (score, true pos., false. pos) for each box
        and a list of the false negatives.
    """
    sim_values = list(similar_classes.values())

    # pre-filter data, remove unknown classes
    pred = filter_data(pred, classes)[0]
    target = filter_data(target, classes + sim_values)[0]

    if bev:
        overlap = iou_bev(pred['bbox'][:, [0, 2, 3, 5, 6]].astype(np.float32),
                          target['bbox'][:, [0, 2, 3, 5, 6]].astype(np.float32))
    else:
        overlap = iou_3d(pred['bbox'].astype(np.float32),
                         target['bbox'].astype(np.float32))

    detection = np.zeros(
        (len(classes), len(difficulties), len(pred['bbox']), 3))
    fns = np.zeros((len(classes), len(difficulties), 1), dtype="int64")
    for i, label in enumerate(classes):
        # filter only with label
        pred_label, pred_idx_l = filter_data(pred, [label])
        target_label, target_idx_l = filter_data(
            target, [label, similar_classes.get(label)])
        overlap_label = overlap[pred_idx_l][:, target_idx_l]
        for j, diff in enumerate(difficulties):
            # filter with difficulty
            pred_idx = filter_data(pred_label, [label], [diff])[1]
            target_idx = filter_data(target_label, [label], [diff])[1]

            if len(pred_idx) > 0:
                # no matching gt box (filtered preds vs all targets)
                fp = np.all(overlap_label[pred_idx] < min_overlap[i],
                            axis=1).astype("float32")

                # identify all matches (filtered preds vs filtered targets)
                match_cond = np.any(
                    overlap_label[pred_idx][:, target_idx] >= min_overlap[i],
                    axis=-1)
                tp = np.zeros((len(pred_idx),))

                # all matches first fp
                fp[np.where(match_cond)] = 1

                # only best match can be tp
                max_idx = np.argmax(overlap_label[:, target_idx], axis=0)
                max_cond = [idx in max_idx for idx in pred_idx]
                match_cond = np.all([max_cond, match_cond], axis=0)
                tp[match_cond] = 1
                fp[match_cond] = 0

                # no matching pred box (all preds vs filtered targets)
                fns[i, j] = np.sum(
                    np.all(overlap_label[:, target_idx] < min_overlap[i],
                           axis=0))
                detection[i, j, [pred_idx]] = np.stack(
                    [pred_label['score'][pred_idx], tp, fp], axis=-1)
            else:
                fns[i, j] = len(target_idx)

    return detection, fns


def sample_thresholds(scores, gt_cnt, sample_cnt=41):
    """Computes equally spaced sample thresholds from given scores

    Args:
        scores (list): list of scores
        gt_cnt (number): amount of gt samples
        sample_cnt (number): amount of samples
            Default is 41.

    Returns:
        Returns a list of equally spaced samples of the input scores.
    """
    scores = np.sort(scores)[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / gt_cnt
        r_recall = (i + 2) / gt_cnt if i < (len(scores) - 1) else l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall)) and
            (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (sample_cnt - 1.0)
    return thresholds


def mAP(pred,
        target,
        classes=[0],
        difficulties=[0],
        min_overlap=[0.5],
        bev=True,
        samples=41,
        similar_classes={}):
    """Computes mAP of the given prediction (11-point interpolation).

    Args:
        pred (dict): List of dictionaries with the prediction data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],
                'score':      [...],
                'difficulty': [...]
            }[]
        target (dict): List of dictionaries with the target data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],
                'difficulty': [...]
            }[]
        classes (number[]): List of classes which should be evaluated.
            Default is [0].
        difficulties (number[]): List of difficulties which should evaluated.
            Default is [0].
        min_overlap (number[]): Minimal overlap required to match bboxes.
            One entry for each class expected. Default is [0.5].
        bev (boolean): Use BEV IoU (else 3D IoU is used).
            Default is True.
        samples (number): Count of used samples for mAP calculation.
            Default is 41.
        similar_classes (dict): Assign classes to similar classes that were not part of the training data so that they are not counted as false negatives.
            Default is {}.

    Returns:
        Returns the mAP for each class and difficulty specified.
    """
    if len(min_overlap) != len(classes):
        assert len(min_overlap) == 1
        min_overlap = min_overlap * len(classes)
    assert len(min_overlap) == len(classes)

    cnt = 0
    box_cnts = [0]
    for p in pred:
        cnt += len(filter_data(p, classes)[1])
        box_cnts.append(cnt)

    gt_cnt = np.zeros((len(classes), len(difficulties)))
    for i, c in enumerate(classes):
        for j, d in enumerate(difficulties):
            for t in target:
                gt_cnt[i, j] += len(filter_data(t, [c], [d])[1])

    detection = np.zeros((len(classes), len(difficulties), box_cnts[-1], 3))
    fns = np.zeros((len(classes), len(difficulties), 1), dtype='int64')
    for i in range(len(pred)):
        d, f = precision_3d(pred=pred[i],
                            target=target[i],
                            classes=classes,
                            difficulties=difficulties,
                            min_overlap=min_overlap,
                            bev=bev,
                            similar_classes=similar_classes)
        detection[:, :, box_cnts[i]:box_cnts[i + 1]] = d
        fns += f

    mAP = np.zeros((len(classes), len(difficulties), 1))
    if samples <= 0:
        # No samples to compute mAP against, so all results are zero.
        return mAP

    for i in range(len(classes)):
        for j in range(len(difficulties)):
            det = detection[i, j, np.argsort(-detection[i, j, :, 0])]

            #gt_cnt = np.sum(det[:,1]) + fns[i, j]
            thresholds = sample_thresholds(det[np.where(det[:, 1] > 0)[0], 0],
                                           gt_cnt[i, j], samples)
            if len(thresholds) == 0:
                # No predictions met cutoff thresholds, skipping AP computation to avoid NaNs.
                continue

            prec = np.zeros((len(thresholds),))
            for ti in range(len(thresholds))[::-1]:
                d = det[np.where(det[:, 0] >= thresholds[ti])]
                tp_acc = np.sum(d[:, 1])
                fp_acc = np.sum(d[:, 2])
                if (tp_acc + fp_acc) > 0:
                    prec[ti] = tp_acc / (tp_acc + fp_acc)
                prec[ti] = np.max(prec[ti:], axis=-1)

            if len(prec[::4]) < int(samples / 4 + 1):
                mAP[i, j] = np.sum(prec) / len(prec) * 100
            else:
                mAP[i, j] = np.sum(prec[::4]) / int(samples / 4 + 1) * 100

    return mAP

def iou_2d(box1, box2):
    """Computes IoU between two bounding boxes"""
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2

    left = max(l1, l2)
    top = max(t1, t2)
    right = min(r1, r2)
    bottom = min(b1, b2)
    if left >= right or top >= bottom:
        return 0.0
    
    area_inter = (right - left) * (bottom - top)
    area_1 = (r1 - l1) * (b1 - t1)
    area_2 = (r2 - l2) * (b2 - t2)
    area_union = area_1 + area_2 - area_inter
    iou = area_inter / area_union
    return iou

def cvt_tensor(pred, target, classes=[0]):
    filtered_pred, _ = filter_data(pred, classes)
    filtered_target, _ = filter_data(target, classes)

    # convert classes to int values
    filtered_pred['label'] = np.array([classes.index(l) for l in filtered_pred['label']])
    filtered_target['label'] = np.array([classes.index(l) for l in filtered_target['label']])

    pred_t = dict(
        boxes=torch.tensor(filtered_pred['bbox']),
        scores=torch.tensor(filtered_pred['score']),
        labels=torch.tensor(filtered_pred['label'])
    )

    target_t = dict(
        boxes=torch.tensor(filtered_target['bbox']),
        labels=torch.tensor(filtered_target['label'])
    )

    return pred_t, target_t

def dist_error_2d(pred, target, min_overlap=0.5, classes=[0], diffs=None):
    """Computes the dis_to_cam error between predicted and ground truth bounding boxes for each class"""
    
    filtered_pred, _ = filter_data(pred, classes)
    filtered_target, _ = filter_data(target, classes, diffs)
    
    pred_dists = []
    gt_dists = []
    difficulties = []
    labels = []

    for p_idx, p_box in enumerate(filtered_pred['bbox']):
        best_iou = 0
        best_t_idx = -1
        
        for t_idx, t_box in enumerate(filtered_target['bbox']):
            current_iou = iou_2d(p_box, t_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_t_idx = t_idx
                
        if best_iou >= min_overlap:
            t_dist = filtered_target['dis_to_cam'][best_t_idx]
            pred_dist = filtered_pred['dis_to_cam'][p_idx]
            label = filtered_pred['label'][p_idx]
            difficulty = filtered_target['difficulty'][best_t_idx]

            pred_dists.append(pred_dist)
            gt_dists.append(t_dist)
            difficulties.append(difficulty)
            labels.append(label)
    
    return np.array(pred_dists), np.array(gt_dists), np.array(difficulties), np.array(labels, dtype='<U20')

def rel_error_2d(pred):
    """Returns an array containing the relative error and dis_to_cam for each prediction"""
    labels = np.unique(pred['label'])
    error_dict = {}

    for label in labels:
        filtered_pred, _ = filter_data(pred, [label])
        errors = []
        dists = []
        for p in filtered_pred:
            if not np.isnan(p['dis_to_cam']) and not np.isnan(p['rel_error']):
                lidar_dist = p['dis_to_cam']
                rel_error = p['rel_error']
                errors.append(rel_error)
                dists.append(lidar_dist)

        errors = np.array(errors)
        dists = np.array(dists)
        error_dict[label] = {'error': errors, 'fusion_dist': dists}
    return error_dict
