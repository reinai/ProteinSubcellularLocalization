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

    figure, axis = plt.subplots(1,4, figsize=(16,4))
    figure.suptitle("Plot images", fontsize=24)
    for index, color in enumerate(["green","blue","yellow","red"]):
        image_filename_path = DS_PATH + "/train/" + image_id + "_" + color + ".png"
        img = plt.imread(image_filename_path)
        axis[index].imshow(img, cmap='binary')
        axis[index].set_title(color)
    plt.show()


def plot_image_rgb(image_id):
    """
    Ploting a protein image in rgb form by combining all 4 color filters into one rgb photo

    :param image_id: id of protein image
    """

    # dimensions of one photo is 512x512 and there are 4 colors
    all_color_images = np.zeros((512,512,4))
    for index, color in enumerate(["green", "blue", "yellow", "red"]):
        image_filename_path = DS_PATH + "/train/" + image_id + "_" + color + ".png"
        all_color_images[:, :, index] = plt.imread(image_filename_path)
    # We get yellow from red and green so there is an extra 1 for red and green to represent that color
    transformation_matrix = np.array([[1,0,0,1],[1,0,1,0],[0,1,0,0]])
    rgb_image = np.matmul(all_color_images.reshape(-1, 4), np.transpose(transformation_matrix))
    rgb_image = rgb_image.reshape(all_color_images.shape[0], all_color_images.shape[0], 3)
    rgb_image = np.clip(rgb_image, 0, 1)
    plt.imshow(rgb_image)
    plt.title("Plot image RGB", fontsize=24)
    plt.show()


def print_number_of_targets_per_label(train):
    """
    Printing a count for each label how much times it appears in protein images and plotting the results

    :param train: train dataframe that contain id for all protein images with targeted labels
    """

    labels_count = np.zeros(len(list_of_labels))
    for image_target in train.Target:
        label_ints = [int(label) for label in image_target.split()]
        labels_count[label_ints] += 1
    # printing counts for each label
    for index, value in enumerate(labels_count):
        print(list_of_labels[index] + " - " + str(value))
    # plotting counts for each label
    plt.figure(figsize=(14, 6))
    plt.title("Count for labels", fontsize=24)
    plt.bar(range(len(list_of_labels)), labels_count)
    plt.ylabel('labels count')
    plt.xticks(range(len(list_of_labels)), list_of_labels, rotation=-90);
    plt.show()


def plot_correlation_matrix(train):
    """
    Plotting a correlation matrix to see is there some connections and dependency between labels of protein

    :param train: train dataframe that contain id for all protein images with targeted labels
    """

    number_of_labels = len(list_of_labels)
    number_of_samples = train.shape[0]
    sample_labels_matrix = np.zeros((number_of_samples, number_of_labels))
    for index, image_target in enumerate(train.Target):
        label_ints = [int(label) for label in image_target.split()]
        sample_labels_matrix[index, label_ints] = 1
    correlation_matrix = np.corrcoef(sample_labels_matrix, rowvar=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, xticklabels=list_of_labels, yticklabels=list_of_labels)
    plt.title('Correlation matrix', fontsize=24)
    plt.show()


def plot_target_size(train):
    """
    Plotting and printing numbers of how many labels are there in photos in percentage and in raw numbers

    :param train: train dataframe that contain id for all protein images with targeted labels
    """

    target_size_per_image = np.zeros(train.shape[0])
    for iter in range(len(target_size_per_image)):
        target_size_per_image[iter] = len(train.Target[iter].split())
    target_counts = np.unique(target_size_per_image, return_counts=True)
    percentage_count = np.round(100 * (np.array(target_counts[1]) / train.shape[0]))
    # printing the target size
    for iter in range(len(target_counts[0])):
        print("Target length " + str(np.array(target_counts[0])[iter]) + " - " + str(np.array(target_counts[1])[iter]))
    # plotting the target size
    plt.figure(figsize=(20, 5))
    plt.title('Target size', fontsize=24)
    sns.barplot(x=target_counts[0], y=percentage_count, palette="Reds")
    plt.xlabel("Number of targets per image")
    plt.ylabel("% of data")
    plt.show()


def plot_specific_label(label, train):
    """
    Plotting the number of occurrence of other labels with some specific label to see some concrete correlation

    :param label: specific label
    :param train: train dataframe that contain id for all protein images with targeted labels
    """

    def count_specific_label(label):
        """
        Counting the number of occurrence other labels with some specific label to see some concrete correlation

        :param label: specific label
        :return: two arrays where first one are correlated labels and second one is number of occurrence of that
                 correlated label
        """

        label_count = []
        for iter in range(train.shape[0]):
            if str(label) in train.Target[iter].split():
                for sample_label in train.Target[iter].split():
                    label_count.append(sample_label)
        return np.unique(np.array(label_count), return_counts=True)

    number_specific_label = count_specific_label(label)
    # plotting the number of occurrence of specific label with other labels
    x_plot_axis = [list_of_labels[int(sample_label)] for sample_label in np.array(number_specific_label[0])]
    y_plot_axis = np.array(number_specific_label[1])
    plt.figure(figsize=(10, 3))
    plt.title(list_of_labels[int(label)] + "  - correlation with other labels", fontsize=24)
    sns.barplot(x=x_plot_axis, y=y_plot_axis, palette="Blues")
    plt.xlabel("Label name")
    plt.ylabel("Number of occurrence with " + list_of_labels[int(label)])
    plt.show()



if __name__ == "__main__":
    """
    Main function to test our functions and explore the data
    """

    print("\n--- PRINT DATASET INFO ---")
    train = read_dataset()
    print("\n--- PRINT LABELS FOR AN IMAGE ---")
    print_image_labels(train.Target[1])
    plot_image_binary(train.Id[1])
    plot_image_rgb(train.Id[1])
    print("\n--- PRINT NUMBER OF TARGETS PER LABELS ---")
    print_number_of_targets_per_label(train)
    plot_correlation_matrix(train)
    print("\n--- PRINT TARGET SIZE ---")
    plot_target_size(train)
    # Lysosoms has an id 10
    plot_specific_label('10', train)
    # Endosomes has an id 9
    plot_specific_label('9', train)
    # Microtubule ends has an id 15
    plot_specific_label('15', train)
    # Peroxisomes has an id 8
    plot_specific_label('8', train)
    # Rods & rings has an id 27
    plot_specific_label('27', train)
