# author: Nikola Zubic

import pandas as pd
import os

DATA_DIRECTORY = "../dataset"
TEST = "/home/nikola/Documents/TEST_PROTEIN/test.csv"
# using a split that includes all classes in val
with open("tr_names.txt", 'r') as text_file:
    tr_n = text_file.read().split(',')
with open("val_names.txt", 'r') as text_file:
    val_n = text_file.read().split(',')
test_names = sorted({f[:36] for f in os.listdir(TEST)})
print(len(tr_n), len(val_n))


class Oversampling:
    def __init__(self, path):
        self.train_labels = pd.read_csv(path).set_index('Id')
        self.train_labels['Target'] = [[int(i) for i in s.split()]
                                       for s in self.train_labels['Target']]
        # set the minimum number of duplicates for each class
        self.multi = [1, 1, 1, 1, 1, 1, 1, 1,
                      4, 4, 4, 1, 1, 1, 1, 4,
                      1, 1, 1, 1, 2, 1, 1, 1,
                      1, 1, 1, 4]

    def get(self, image_id):
        labels = self.train_labels.loc[image_id, 'Target'] if image_id \
                                                              in self.train_labels.index else []
        m = 1
        for l in labels:
            if m < self.multi[l]: m = self.multi[l]
        return m


s = Oversampling(os.path.join(DATA_DIRECTORY, "/train.csv"))
tr_n = [idx for idx in tr_n for _ in range(s.get(idx))]
print(len(tr_n), flush=True)  # 29016
