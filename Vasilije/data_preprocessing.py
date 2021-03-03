import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage.filters
import scipy.ndimage
import sklearn.feature_extraction
import sklearn.cluster
import skimage.feature
import skimage.transform
import PIL
plt.gray()

DS_PATH = "../dataset/train/"


def plot_image(image_filename):
    """
    Plotting an image based on its filename and plotting a histogram that represents the amount of each value of gray
    the image consists of

    :param image_filename: filename of the image
    """

    image = plt.imread(DS_PATH + image_filename)
    histogram = np.histogram(image - image.mean(), bins=np.arange(image.min(), image.max(), 1/256))
    figure, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, interpolation='nearest')
    axes[0].axis('off')
    axes[0].set_title(image_filename[-20:])
    axes[1].plot(histogram[1][:-1], histogram[0], lw=2)
    axes[1].set_title('histogram of gray values')
    plt.show()


def otsu_thresholding(image_filename):
    """
    Otsu thresholding is a technique that is used for image thresholding to separate the pixels of the image into two
    classes: foreground and background. This method finds the threshold automatically.

    :param image_filename: filename of the image
    """

    def set_title(title, threshold):
        if np.shape(threshold) != ():
            plt.title('%s threashold=[%s]' % (title, str(np.shape(threshold))))
        else:
            plt.title('%s threashold=%0.3f' % (title, threshold))

    fig = plt.figure(figsize=(15, 4))
    fig.suptitle("Otsu threshold", fontsize=24)
    for index, color in enumerate(['red','green','blue','yellow']):
        current_color_image_path = DS_PATH + image_filename + '_' + color + '.png'
        plt.subplot(1, 4, index+1)
        current_image = plt.imread(current_color_image_path)
        threshold = skimage.filters.threshold_otsu(current_image)
        plt.imshow(current_image > threshold)
        set_title(color, threshold)
    plt.show()


def convert_to_binary_image(image_filename):
    """
    COnverting an image into a binary photo using li threshold

    :param image_filename: filename of the image
    :return: a binary photo
    """

    current_image = plt.imread(DS_PATH + image_filename)
    return np.array(current_image > skimage.filters.threshold_li(current_image))


def image_reduction(image_filename):
    """
    Image reduction is a technique to make an image smaller without losing any information about it, by making one 8x8
    block of pixels into 1 pixel by concatenating all of 64 pixels into one big string

    :param image_filename: filename of the image
    """

    current_image = plt.imread(DS_PATH + image_filename)
    current_binary_image = convert_to_binary_image(image_filename)
    x_resolution, y_resolution = np.shape(current_image)
    x_resolution = int(x_resolution / 8)
    y_resolution = int(y_resolution / 8)
    reduced_image = np.empty((x_resolution, y_resolution))
    for x in range(x_resolution):
        for y in range(y_resolution):
            reduced_image[x][y] = int("".join(current_binary_image[x * 8:(x * 8) + 8, y * 8:(y * 8) + 8].flatten()
                                              .astype(int).astype(str)), 2)
    # plotting the reduced image
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle("Image reduction", fontsize=24)
    for index, list_title_image in enumerate([['original', current_image], ['binary', current_binary_image],
                                              ['compressed', reduced_image]]):
        plt.subplot(1, 3, index+1)
        plt.title(list_title_image[0])
        plt.imshow(list_title_image[1])
    plt.show()


def binary_opening(image_filename):
    """
    Binary opening is a technique to do the dilation (spreading pixels on the edges) of the erosion ( closing pixels
    on the edges)

    :param image_filename: filename of the image
    """

    current_image = plt.imread(DS_PATH + image_filename)
    current_binary_image = convert_to_binary_image(image_filename)
    current_binary_opening_photo = scipy.ndimage.binary_opening(current_binary_image)
    # plotting the binary opening
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle("Binary opening", fontsize=24)
    for index, list_title_image in enumerate([['original', current_image], ['binary', current_binary_image],
                                              ['binary opening', current_binary_opening_photo]]):
        plt.subplot(1, 3, index+1)
        plt.title(list_title_image[0])
        plt.imshow(list_title_image[1])
    plt.show()


def image_segmentation(image_filename):
    """
    Image segmentation is a technique to represent a photo into more meaningful parts that are easier to analyze

    :param image_filename:  filename of the image
    """

    current_image = plt.imread(DS_PATH + image_filename)
    label_image, _ = scipy.ndimage.label(current_image > current_image.mean())
    plt.imshow(label_image, cmap='nipy_spectral')
    print('Number of unique segments - ', np.unique(label_image).size)
    plt.show()


if __name__ == "__main__":
    """
    Main function to test our functions and preprocess the data
    """

    image_filename = '000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0'
    for color in ['red','green','blue','yellow']:
        plot_image(image_filename + '_' + color + '.png')
    otsu_thresholding(image_filename)
    image_reduction(image_filename + '_' + "yellow" + '.png')
    binary_opening(image_filename + '_' + "yellow" + '.png')
    image_segmentation(image_filename + '_' + "green" + '.png')
