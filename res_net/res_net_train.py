# author: Nikola Zubic

from metrics import F1_callback, FocalLoss
from network_architecture import ResNet
from fastai.conv_learner import *
from fastai.dataset import *
from utils import get_data

image_size = 512
batch_size = 64


def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()


md = get_data(image_size, batch_size)
learner = ConvLearner.pretrained(ResNet, md, ps=0.5)  # dropout 50%
learner.opt_fn = optim.Adam

learner.clip = 1.0  # gradient clipping

learner.crit = FocalLoss()
f1_callback = F1_callback()
learner.metrics = [acc, f1_callback.f1]

"""
We begin by finding the optimal learning rate. The following function runs training with different learning rates and 
records the loss. Increase of the loss indicates that we are not doing something right - divergence of training. The 
optimal lr lies in the minimum part of the curve, but before the onset of divergence. Based on the following plot, 
for the current setup the divergence starts at ~0.05, and the recommended learning rate is ~0.005.
"""

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    learner.lr_find()
learner.sched.plot()

"""
First, we train only the head of the model while keeping the rest frozen. It allows to avoid corruption of the 
pretrained weights at the initial stage of training due to random initialization of the head layers. So the power of 
transfer learning is fully utilized when the training is continued.
"""

lr = 0.5e-2
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    learner.fit(lr, 1, callbacks=[f1_callback])

"""
Next, we unfreeze all the weights and allow training of entire model. One trick that we use is differential learning 
rate: the lr of the head part is still lr, while the middle layers of the model are trained with lr / 3, and the base is 
trained with even smaller lr / 10. Despite the low-level detectors do not vary too much, the yellow channel should be 
trained, and also the images are quite different from ImageNet. So, we decrease the learning rate for first layers only 
by 10 times. If there was no necessity to train an additional channel and the images were more similar to ImageNet, the 
learning rates could be [lr / 100,lr / 10, lr]. Another trick is learning rate annealing. Periodic lr increase followed 
by slow decrease drives the system out of steep minimum (when lr is high) towards broader ones (which are explored when 
lr decreases) that enhances the ability of the model to generalize and reduces over-fitting. The length of the cycles 
gradually increases during training.
"""
learner.unfreeze()
lrs = np.array([lr / 10, lr / 3, lr])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    learner.fit(lrs / 4, 4, cycle_len=2, use_clr=(10, 20), callbacks=[f1_callback])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    learner.fit(lrs / 16, 2, cycle_len=4, use_clr=(10, 20), callbacks=[f1_callback])
