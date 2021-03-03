import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DS_PATH = "../dataset"

list_of_labels = [
    'Nucleoplasm',                   # 0
    'Nuclear membrane',              # 1
    'Nucleoli',                      # 2
    'Nucleoli fibrillar center',     # 3
    'Nuclear speckles',              # 4
    'Nuclear bodies',                # 5
    'Endoplasmic reticulum',         # 6
    'Golgi apparatus',               # 7
    'Peroxisomes',                   # 8
    'Endosomes',                     # 9
    'Lysosomes',                     # 10
    'Intermediate filaments',        # 11
    'Actin filaments',               # 12
    'Focal adhesion sites',          # 13
    'Microtubules',                  # 14
    'Microtubule ends',              # 15
    'Cytokinetic bridge',            # 16
    'Mitotic spindle',               # 17
    'Microtubule organizing center', # 18
    'Centrosome',                    # 19
    'Lipid droplets',                # 20
    'Plasma membrane',               # 21
    'Cell junctions',                # 22
    'Mitochondria',                  # 23
    'Aggresome',                     # 24
    'Cytosol',                       # 25
    'Cytoplasmic bodies',            # 26
    'Rods & rings'                   # 27
]

def read_dataset():
    """
    Reading train set based on path we defined as a constant variable and providing
    small info and head of that train dataset

    :return: train dataset
    """

    train_set = pd.read_csv(DS_PATH + "/train.csv")
    print("Train set info:")
    print(train_set.info())
    print("Head of train set: ")
    print(train_set.head(5))
    return train_set


def print_image_labels(image_labels):
    """
    Printing all of image labels in format index - full name of label

    :param image_labels: string of labels for concrete image
    """

    list_of_image_labels = [int(label) for label in image_labels.split()]
    for label in list_of_image_labels:
        print(str(label) + " - " + list_of_labels[label])


def plot_image_binary(image_id):
    """
    Ploting a protein image in binary form for all four color filters

    :param image_id: id of protein image
    """

    fig, axis = plt.subplots(1,4, figsize=(16,4))
    for index, color in enumerate(["green","blue","yellow","red"]):
        image_filename_path = DS_PATH + "/train/" + image_id + "_" + color + ".png"
        img = plt.imread(image_filename_path)
        axis[index].imshow(img, cmap='binary')
        axis[index].set_title(color)
    plt.show()


if __name__ == "__main__":
    """
    Main function to test our function and explore the data
    """

    print("\n--- PRINT DATASET INFO ---")
    train = read_dataset()
    print("\n--- PRINT LABELS FOR AN IMAGE ---")
    print_image_labels(train.Target[1])
    plot_image_binary(train.Id[1])