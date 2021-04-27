# author: Nikola Zubic

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from keras.callbacks import Callback


class FocalLoss(nn.Module):
    # In order to address the issue of strong data imbalance, we use the Focal Loss function.
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class F1(object):
    def __init__(self, n=28):
        self.n = n
        self.TP = np.zeros(self.n)
        self.FP = np.zeros(self.n)
        self.FN = np.zeros(self.n)

    def __call__(self,preds,targs,th=0.0):
        preds = (preds > th).int()
        targs = targs.int()
        self.TP += (preds*targs).float().sum(dim=0)
        self.FP += (preds > targs).float().sum(dim=0)
        self.FN += (preds < targs).float().sum(dim=0)
        score = (2.0*self.TP/(2.0*self.TP + self.FP + self.FN + 1e-6)).mean()
        return score

    def reset(self):
        #macro F1 score
        score = (2.0 * self.TP / (2.0 * self.TP + self.FP + self.FN + 1e-6))
        print('F1 macro:', score.mean(), flush=True)


        self.TP = np.zeros(self.n)
        self.FP = np.zeros(self.n)
        self.FN = np.zeros(self.n)


class F1_callback(Callback):
    def __init__(self, n=28):
        super().__init__()
        self.f1 = F1(n)

    def on_epoch_end(self, metrics):
        self.f1.reset()
