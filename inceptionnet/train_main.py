from inception_nn_model import CustomInceptionModel
import tensorflow as tf

if __name__ == "__main__":
    """
    Main function to train our Inception model
    """

    model = CustomInceptionModel(input_shape=(299, 299, 3),
                                 output_size=28,
                                 warmup_epochs=2,
                                 regular_epochs=50,
                                 batch_size=16,
                                 checkpoint_path='./checkpoint/InceptionV3.h5',
                                 path_to_train="../dataset",
                                 path_to_test="../dataset",
                                 image_size=299)
    model.train_inception_model()
    """
    model.create_inception_model()
    dot_img_file = './model_image/custom_inceptionV3_model.png'
    tf.keras.utils.plot_model(model.model, to_file=dot_img_file, show_shapes=True)
    """
