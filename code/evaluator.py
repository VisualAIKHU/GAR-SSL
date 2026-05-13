import numpy as np


class Evaluator:
    def __init__(self):
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.5):
        infer_map = np.zeros_like(infer)
        infer_map[infer >= thres] = 1
        ciou = np.sum(infer_map * gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap == 0)) + 1e-8)
        self.ciou.append(ciou)
        return ciou

    def final(self):
        return np.mean(self.ciou) if self.ciou else 0.0

    def clear(self):
        self.ciou = []
