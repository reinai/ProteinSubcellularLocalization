# author: Nikola Zubic

import cv2
import numpy as np
import os
import pandas as pd
from label_information import label_names
from fastai.transforms import RandomRotate, TfmType, RandomDihedral, RandomLighting, Cutout, CropType
from fastai.dataset import FilesDataset
from oversampling import TEST, DATA_DIRECTORY


def read_rgby_image(path, id):
    colors = ['red', 'green', 'blue', 'yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id + '_' + color + '.png'), flags).astype(np.float32) / 255
           for color in colors]
    return np.stack(img, axis=-1)


class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.labels = pd.read_csv(DATA_DIRECTORY + "/train.csv").set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        return read_rgby_image(self.path, self.fnames[i])

    def get_y(self, i):
        if self.path == TEST:
            return np.zeros(len(label_names), dtype=np.int)
        else:
            labels = self.labels.loc[self.fnames[i]]['Target']
            return np.eye(len(label_names), dtype=np.float)[labels].sum(axis=0)

    @property
    def is_multi(self):
        return True

    @property
    def is_reg(self):
        return True

    """
    This flag is set to remove the output sigmoid that allows log(sigmoid) optimization of the numerical stability of 
    the loss function.
    """
    def get_c(self):
        return len(label_names)  # number of classes


# used fast.ai
def get_data(sz, bs, is_test=False):
    #data augmentation
    if is_test:
        aug_tfms = [RandomRotate(30, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO)]
    else:
        aug_tfms = [RandomRotate(30, tfm_y=TfmType.NO), RandomDihedral(tfm_y=TfmType.NO),
                    RandomLighting(0.05, 0.05, tfm_y=TfmType.NO), Cutout(n_holes=25, length=10*sz//128,
                                                                         tfm_y=TfmType.NO)]
    #mean and std in of each channel in the train set
    stats = A([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)], TRAIN),
                (val_n,TRAIN), tfms, test=(test_names, TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    return md
