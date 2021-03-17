import matplotlib.pyplot as plt
import skimage.filters
import scipy.ndimage
import skimage.feature
import skimage.transform
import numpy as np
from PIL import Image
import cv2
import matplotlib.image as mpimg
from glob import glob
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import spectral_clustering as clustering_sklearn
from skimage.morphology import watershed
from albumentations import (
    HorizontalFlip, Blur, RandomGamma, ElasticTransform, ChannelShuffle, Rotate
)
plt.gray()

DATASET_PATH = "../dataset/train/"


def yen_thresholding(img_file_name):
    """
    Thresholding produces a specific black and white representation of the gray-scale image.
    We can use this as a compressed input.

    :param img_file_name: given file name
    :return: None, shows the plot
    """

    def set_title(title, threshold):
        if np.shape(threshold) != ():
            plt.title('%s threshold=[%s]' % (title, str(np.shape(threshold))))
        else:
            plt.title('%s threshold=%0.3f' % (title, threshold))

    figure = plt.figure(figsize=(15, 4))
    figure.suptitle("Yen threshold", fontsize=24)

    for index, color in enumerate(['red', 'green', 'blue', 'yellow']):
        current_color_image_path = DATASET_PATH + img_file_name + '_' + color + '.png'
        plt.subplot(1, 4, index+1)

        current_image = plt.imread(current_color_image_path)
        threshold = skimage.filters.threshold_yen(current_image)

        plt.imshow(current_image > threshold)
        set_title(color, threshold)

    plt.show()


def image_compression(img_file_name):
    """
    Compresses an image. Useful for faster training while still achieving similar results.

    :param img_file_name: image file name useful for compression
    :return: saves compressed images at compressed_images_example folder and plots difference
    """

    for _, color in enumerate(['red', 'green', 'blue', 'yellow']):
        image = Image.open(DATASET_PATH + img_file_name + '_' + color + '.png')

        image.save("compressed_images_example/original/" + img_file_name + '_' + color + "_original.png")

        # Image downsize with antialiasing filter
        new_image = image.resize((128, 128), Image.ANTIALIAS)
        new_image.save("compressed_images_example/" + img_file_name + '_' + color + '.png', quality=95)

        f = plt.figure()
        plt.title("Before (left) and after (right) the image compression")
        f.add_subplot(1, 2, 1)
        plt.imshow(np.rot90(image, 2))
        f.add_subplot(1, 2, 2)
        plt.imshow(np.rot90(new_image, 2))
        plt.show(block=True)


def convert_to_binary_image(img_file_name):
    """
    Converts an image into a binary image by using Li threshold.

    :param img_file_name: image file name useful for binarization
    :return: a binary photo
    """

    current_image = plt.imread(DATASET_PATH + img_file_name)
    return np.array(current_image > skimage.filters.threshold_li(current_image))


def binary_closing(image_filename):
    """
    Multidimensional binary closing with the given structuring element.
    The closing of an input image by a structuring element is the erosion of the dilation of the image by the
    structuring element.

    :param image_filename: image file name useful for binary closing
    """
    current_image = plt.imread(DATASET_PATH + image_filename)
    current_binary_image = convert_to_binary_image(image_filename)
    current_binary_opening_photo = scipy.ndimage.binary_closing(current_binary_image)

    # plotting the binary closing
    figure = plt.figure(figsize=(15, 4))
    figure.suptitle("Binary closing", fontsize=24)
    for index, list_title_image in enumerate([['original', current_image], ['binary', current_binary_image],
                                              ['binary closing', current_binary_opening_photo]]):
        plt.subplot(1, 3, index+1)
        plt.title(list_title_image[0])
        plt.imshow(list_title_image[1])
    plt.show()


def spectral_clustering(image_filename):
    """
    Performs a Spectral Clustering for a certain image.

    :param image_filename: image file name useful for Spectral Clustering
    :return: processed image
    """
    image = plt.imread(DATASET_PATH + image_filename)

    cutted = image > image.mean()

    graph = img_to_graph(image, mask=cutted)

    graph.data = np.exp(-graph.data / graph.data.std())

    labels = clustering_sklearn(graph, n_clusters=5, eigen_solver='arpack')

    label_im = -np.ones(cutted.shape)
    label_im[cutted] = labels

    plt.imshow(label_im, cmap='nipy_spectral')
    plt.show()


def contour_selection(image_filename):
    """
    Contour selection for an image, useful as an additional feature while training the model.

    :param image_filename: image file name useful for Contour Selection
    :return: processed image
    """
    image = cv2.imread(DATASET_PATH + image_filename, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    thresh_img = cv2.adaptiveThreshold(gray, 512, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    plt.imshow(thresh_img, cmap="gray")
    plt.title("Contour selection")
    plt.show()


def blob_detection(image_filename):
    """
    In computer vision, blob detection methods are aimed at detecting regions in a digital image that differ in
    properties, such as brightness or color, compared to surrounding regions. Informally, a blob is a region of an image
    in which some properties are constant or approximately constant; all the points in a blob can be considered in some
    sense to be similar to each other. The most common method for blob detection is convolution.
    Patterns of the blobs can be used as an additional feature.

    :param image_filename: image file name useful for Blob detection
    :return: None, shows image after blob detection
    """
    image = plt.imread(DATASET_PATH + image_filename)

    blobs_log = skimage.feature.blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * (2 ** .5)

    blobs_dog = skimage.feature.blob_dog(image, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * (2 ** .5)

    blobs_doh = skimage.feature.blob_doh(image.astype('float64'), max_sigma=30, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']

    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)

    plt.tight_layout()
    plt.show()


def region_based_segmentation(image_filename):
    """
    Region-based segmentation is a technique for determining the region directly. Region growing methods can provide the
    original images which have clear edges with good segmentation results. Useful as additional feature while training
    the model.

    :param image_filename: image file name useful for Blob detection
    :return: None, shows the results
    """
    image = plt.imread(DATASET_PATH + image_filename) * 255

    markers = np.zeros_like(image)
    markers[image > 0] = 1
    markers[image > 25] = 2
    markers[image > 30] = 3
    markers[image > 35] = 4

    plt.imshow(markers, cmap='nipy_spectral')
    plt.title('Segmentation markers')
    plt.show()

    elevation_map = skimage.filters.sobel(image / 255)
    segmentation = watershed(elevation_map, markers)

    plt.imshow(segmentation)
    plt.title('Region-based segmentation')
    plt.show()


def sobel_operator(image_filename):
    """
    The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter, is used in image processing and
    computer vision, particularly within edge detection algorithms where it creates an image emphasising edges.
    Emphasized edges can be used as an additional feature.

    :param image_filename: image file name useful for Sobel operator
    :return: None, shows image after applying the Sobel operator
    """
    image = plt.imread(DATASET_PATH + image_filename) * 255
    image = image.astype(int)

    elevation_map = skimage.filters.sobel(image / 255)

    plt.title('Sobel operator')
    plt.imshow(elevation_map)
    plt.show()


def show_image(image_file):
    red = np.array(Image.open(DATASET_PATH + image_file + "_red.png").convert("L"))
    green = np.array(Image.open(DATASET_PATH + image_file + "_green.png").convert("L"))
    blue = np.array(Image.open(DATASET_PATH + image_file + "_blue.png").convert("L"))
    yellow = np.array(Image.open(DATASET_PATH + image_file + "_yellow.png").convert("L"))

    demo_rgb = Image.fromarray(
        np.concatenate((np.expand_dims(red, axis=2), np.expand_dims(green, axis=2), np.expand_dims(blue, axis=2)),
                       axis=2))
    demo_y = Image.fromarray(
        np.concatenate((np.expand_dims(yellow, axis=2), np.expand_dims(yellow, axis=2), np.expand_dims(blue, axis=2)),
                       axis=2))

    plt.title("4-channel input")
    plt.imshow(demo_y)
    plt.show()

    plt.title("RGB")
    plt.imshow(demo_rgb)
    plt.show()


def image_augmentation(image_file):
    def augment(aug, image):
        return aug(image=image)['image']

    def augment_4chan(aug, image):
        image[:, :, 0:3] = aug(image=image[:, :, 0:3])['image']
        image[:, :, 3] = aug(image=image[:, :, 1:4])['image'][:, :, 2]
        return image

    red = np.array(Image.open(DATASET_PATH + image_file + "_red.png").convert("L"))
    green = np.array(Image.open(DATASET_PATH + image_file + "_green.png").convert("L"))
    blue = np.array(Image.open(DATASET_PATH + image_file + "_blue.png").convert("L"))
    yellow = np.array(Image.open(DATASET_PATH + image_file + "_yellow.png").convert("L"))

    demo_rgb = Image.fromarray(
        np.concatenate((np.expand_dims(red, axis=2), np.expand_dims(green, axis=2), np.expand_dims(blue, axis=2)),
                       axis=2))

    aug = HorizontalFlip(p=1)
    image = Image.fromarray(augment(aug, np.array(demo_rgb)))

    plt.title("Horizontal Flip")
    plt.imshow(image)
    plt.show()

    aug = Blur(p=1, blur_limit=3)
    image = Image.fromarray(augment(aug, np.array(demo_rgb)))

    plt.title("Blur")
    plt.imshow(image)
    plt.show()

    aug = ElasticTransform(p=1, border_mode=cv2.BORDER_REFLECT_101, alpha_affine=40)
    image = Image.fromarray(augment(aug, np.array(demo_rgb)))

    plt.title("Elastic Transform")
    plt.imshow(image)
    plt.show()

    aug = RandomGamma(p=1)
    image = Image.fromarray(augment(aug, np.array(demo_rgb)))

    plt.title("Random Gamma")
    plt.imshow(image)
    plt.show()

    aug = ChannelShuffle(p=1)
    image = Image.fromarray(augment(aug, np.array(demo_rgb)))

    plt.title("Channel Shuffle")
    plt.imshow(image)
    plt.show()

    aug = Rotate(p=1, limit=30)
    image = Image.fromarray(augment(aug, np.array(demo_rgb)))

    plt.title("Image rotation")
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    image_file = "0047c984-bba6-11e8-b2ba-ac1f6b6435d0"

    show_image(image_file)
    image_augmentation(image_file)
    yen_thresholding(image_file)
    image_compression(image_file)
    binary_closing(image_file + '_' + "red" + '.png')
    contour_selection(image_file + '_' + "blue" + '.png')
    blob_detection(image_file + '_' + "yellow" + '.png')
    region_based_segmentation(image_file + '_' + "red" + '.png')
    sobel_operator(image_file + '_' + "green" + '.png')
    #spectral_clustering(image_file + '_' + "green" + '.png')
