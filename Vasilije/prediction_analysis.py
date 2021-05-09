import pandas as pd
import numpy as np
import os

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


def read_labels_from_csv(csv_path):
    """
    Reading labels (outputs) from the csv

    :param csv_path: path to the csv folder with labels and images id's
    :return: array of labels
    """

    labels_from_data = []
    data = pd.read_csv(csv_path)
    for name, labels in zip(data['Id'], data['Predicted'].str.split(' ')):
        # because there are rows that have predicted 0 labels, it is needed to do this part like this
        try:
            labels_new = np.array([int(label) for label in labels])
        except:
            # there is no predicted label
            labels_new = []
        labels_from_data.append(labels_new)
    return labels_from_data


def print_analysis_for_one_label(label):
    """
    Analyzing label predictions and true ground labels for one concrete label by counting how many is correct predicted
    and how many isn't

    :param label: integer number of the label
    """

    predicted_labels = read_labels_from_csv("/home/eugen/Desktop/SIAP/ProteinSubcellularLocalization/models_weights_predictions/inception_net/predicted_InceptionV3.csv")
    true_labels = read_labels_from_csv("/home/eugen/Desktop/SIAP/ProteinSubcellularLocalization/dataset/test.csv")

    label_information = {
        "count": 0,
        "predicted_correct": 0,
        "predicted_wrong": 0
    }

    for index, image_labels in enumerate(true_labels):
        if label in image_labels:
            label_information["count"] = label_information["count"] + 1
            if label in predicted_labels[index]:
                label_information["predicted_correct"] = label_information["predicted_correct"] + 1
            else:
                label_information["predicted_wrong"] = label_information["predicted_wrong"] + 1

    title_string = "= ANALYSIS FOR " + list_of_labels[label].upper() + " LABEL ="
    print("=" * len(title_string))
    print(title_string)
    print("= count -> " + str(label_information["count"]))
    print("= predicted correct -> " + str(label_information["predicted_correct"]))
    print("= predicted wrong -> " + str(label_information["predicted_wrong"]))
    print("=" * len(title_string) + "\n")


if __name__ == "__main__":
    """
    Main function to test our functions and analyze predictions
    """

    for iter in range(28):
        print_analysis_for_one_label(iter)
