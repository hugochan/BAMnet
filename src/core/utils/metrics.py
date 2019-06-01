'''
Created on Oct, 2017

@author: hugo

Note: Modified the official evaluation script provided by Berant et al.
(https://github.com/percyliang/sempre/blob/master/scripts/evaluation.py)
'''
from .generic_utils import normalize_answer


def calc_f1(gold_list, pred_list):
    """Return a tuple with recall, precision, and f1 for one example"""

    # Assume all questions have at least one answer
    if len(gold_list) == 0:
        raise RuntimeError('Gold list may not be empty')
    # If we return an empty list recall is zero and precision is one
    if len(pred_list) == 0:
        return (0, 1, 0)
    # It is guaranteed now that both lists are not empty

    # Normalize answers
    gold_list = [normalize_answer(s) for s in gold_list]
    pred_list = [normalize_answer(s) for s in pred_list]

    precision = 0
    for entity in pred_list:
        if entity in gold_list:
            precision += 1
    precision = float(precision) / len(pred_list)

    recall = 0
    for entity in gold_list:
        if entity in pred_list:
              recall += 1
    recall = float(recall) / len(gold_list)

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * recall * precision / (precision + recall)
    return (recall, precision, f1)

def calc_avg_f1(gold_list, pred_list, verbose=True):
    """Go over all examples and compute recall, precision and F1"""
    avg_recall = 0
    avg_precision = 0
    avg_f1 = 0
    count = 0

    out_f = open('error_analysis.txt', 'w')
    assert len(gold_list) == len(pred_list)
    for i, gold in enumerate(gold_list):
        recall, precision, f1 = calc_f1(gold, pred_list[i])
        avg_recall += recall
        avg_precision += precision
        avg_f1 += f1
        count += 1
        if True:
        # if f1 < 0.6:
            out_f.write('{}\t{}\t{}\t{}\n'.format(i, gold, pred_list[i], f1))
    out_f.close()

    avg_recall = float(avg_recall) / count
    avg_precision = float(avg_precision) / count
    avg_f1 = float(avg_f1) / count
    avg_new_f1 = 0
    if avg_precision + avg_recall > 0:
        avg_new_f1 = 2 * avg_recall * avg_precision / (avg_precision + avg_recall)

    if verbose:
        print("Number of questions: " + str(count))
        print("Average recall over questions: " + str(avg_recall))
        print("Average precision over questions: " + str(avg_precision))
        print("Average f1 over questions: " + str(avg_f1))
        # print("F1 of average recall and average precision: " + str(avg_new_f1))
    return count, avg_recall, avg_precision, avg_f1
