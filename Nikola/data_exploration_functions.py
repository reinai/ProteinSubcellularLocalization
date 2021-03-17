import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imageio import imread


output_label_names = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings"
}


def get_train_dataset():
    """
    Reads train dataset.

    :return: train dataset
    """
    train_set_info = pd.read_csv("../dataset/train.csv")

    return train_set_info


def binary_target_values(example):
    """
    Transforms data table

    :param example: given row
    :return: table where each row has an id, and columns for every output label name (1 if it is output, 0
             otherwise)
    """
    exec("output_label_names")
    example.Target = np.array(example.Target.split(" "))\
        .astype(np.int)

    for certain_number in example.Target:
        organelle_name = output_label_names[int(certain_number)]
        example.loc[organelle_name] = 1

    return example


def correlation_matrix(train_set_info):
    print('''1. Many organelles have weak correlations. Some of them have slightly stronger correlations (such as 
endosomes and lysosomes which are located in endoplasmatic reticulum).''')
    reversed_output_label_names = dict((v, k) for k, v in output_label_names.items())

    number_of_labels = len(reversed_output_label_names)

    number_of_samples = train_set_info.shape[0]

    sample_labels_matrix = np.zeros((number_of_samples, number_of_labels))

    for index, image_target in enumerate(train_set_info.Target):
        label_ints = [int(label) for label in image_target.split()]
        sample_labels_matrix[index, label_ints] = 1

    correlation_matrix = np.corrcoef(sample_labels_matrix, rowvar=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, xticklabels=reversed_output_label_names, yticklabels=reversed_output_label_names,
                cmap="summer")

    plt.title('Correlation matrix', fontsize=24)
    plt.show()


def organelle_count(train_set_info):
    """
    :param train_set_info: given train set info (.csv)
    :return: None, prints information about number of organelles that occur the most
    """

    organelle_count = train_set_info.drop(["Id", "Target"], axis=1) \
        .sum(axis=0) \
        .sort_values(ascending=False)

    print("\n2.) What parts of cell (organelles) occur the most?")
    print('''
    We can observe that the larger and more vital parts of the cell occur more in the training set images. Most 
    of it is the nucleo-plasm and cytoplasm (fluids that fill the cell nucleus and cell contents, respectively), the 
    cell nucleoli, cytosols, and plasma membranes. Also, we can see that this dataset is significantly imbalanced (class 
    imbalance problem).
    ''')

    sns.barplot(x=organelle_count.values, y=organelle_count.index.values, order=organelle_count.index)
    plt.title("What parts of cell (organelles) occur the most?")
    plt.xlabel("Number of occurrences for a certain organelle")
    plt.ylabel("Organelle (output) name")
    plt.show()


def number_of_organelles_per_image(train_set_info):
    """
    :param train_set_info: given train set info (.csv)
    :return: None, prints information about average number of organelles per image
    """
    print("\n3.) How many organelles (outputs) on average do we have per image?")
    print('''Mostly, we have only 1 or 2 organelles per image.''')

    train_set_info["number_of_organelle_outputs"] = train_set_info.drop(["Id", "Target"], axis=1).sum(axis=1)

    count_fraction = np.round(train_set_info["number_of_organelle_outputs"].value_counts() / train_set_info.shape[0], 2)
    sns.barplot(x=count_fraction.index.values, y=count_fraction.values)
    plt.title("How many organelles (outputs) on average do we have per image?")
    plt.xlabel("Number of organelles per image")
    plt.ylabel("Fraction of train dataset")
    plt.show()


class ImageDataLoader(object):
    def __init__(self, bs, target_names):
        """
        :param bs: batch size / number of images to visualize
        :param target_names: names of organelles to show

        """
        self.bs = bs
        self.target_names = target_names

        self.batch_shape = (bs, 4, 512, 512)
        self.train_dataset_path = "../dataset/train/"

        reversed_output_label_names = dict((v, k) for k, v in output_label_names.items())

        self.target_list = [reversed_output_label_names[key] for key in target_names]

        self.images_identifier = None

    def load_image(self, id):
        """
        Loads an image by its id.

        :param id: image id
        :return: image (all filters)
        """
        images = np.zeros(shape=(4, 512, 512))

        images[0, :, :] = imread(self.train_dataset_path + id + "_green" + ".png")
        images[1, :, :] = imread(self.train_dataset_path + id + "_red" + ".png")
        images[2, :, :] = imread(self.train_dataset_path + id + "_blue" + ".png")
        images[3, :, :] = imread(self.train_dataset_path + id + "_yellow" + ".png")

        return images

    @staticmethod
    def images_row(image, y_sub_axis, title=""):
        y_sub_axis[0].imshow(image[0], cmap="Greens")

        y_sub_axis[1].imshow(image[1], cmap="Reds")

        if title != "":
            y_sub_axis[1].set_title("Microtubules Filter")

        y_sub_axis[2].imshow(image[2], cmap="Blues")

        if title != "":
            y_sub_axis[2].set_title("Nucleus filter")

        y_sub_axis[3].imshow(image[3], cmap="Oranges")

        if title != "":
            y_sub_axis[3].set_title("Endoplasmatic Reticulum filter")

        if title != "":
            y_sub_axis[0].set_title(title)

        return y_sub_axis

    @staticmethod
    def create_title(file_id, train_set_info):
        file_targets = train_set_info.loc[train_set_info.Id == file_id, "Target"].values[0]
        title = " - "
        for n in file_targets:
            title += output_label_names[n] + " - "
        return title

    def check_subset(self, targets):
        return np.where(set(self.target_list).issubset(set(targets)), 1, 0)

    def find_matching_data_entries(self, train_set_info):
        train_set_info["check_col"] = train_set_info.Target.apply(
            lambda l: self.check_subset(l)
        )

        self.images_identifier = train_set_info[train_set_info.check_col == 1].Id.values
        train_set_info.drop("check_col", axis=1, inplace=True)

    def get_loader(self):
        filenames = []
        idx = 0
        images = np.zeros(self.batch_shape)
        for img_id in self.images_identifier:
            images[idx, :, :, :] = self.load_image(img_id)
            filenames.append(img_id)
            idx += 1
            if idx == self.batch_shape[0]:
                yield filenames, images
                filenames = []
                images = np.zeros(self.batch_shape)
                idx = 0
        if idx > 0:
            yield filenames, images

    @staticmethod
    def run(train_set_info, organelles_of_interest, batch_size=5):
        imageloader = ImageDataLoader(batch_size, organelles_of_interest)
        imageloader.find_matching_data_entries(train_set_info)
        iterator = imageloader.get_loader()
        file_ids, images = next(iterator)

        fig, ax = plt.subplots(len(file_ids), 4, figsize=(30, 5 * len(file_ids)))
        if ax.shape == (4,):
            ax = ax.reshape(1, -1)

        do = 0

        for n in range(len(file_ids)):
            if do == 0:
                ImageDataLoader.images_row(images[n], ax[n],
                                           ImageDataLoader.create_title(file_ids[n], train_set_info, ))
            else:
                ImageDataLoader.images_row(images[n], ax[n])

            do += 1

        plt.show()
