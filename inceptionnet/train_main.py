from inception_nn_model import CustomInceptionModel

if __name__ == "__main__":
    """
    Main function to train our Inception model
    """

    model = CustomInceptionModel(input_shape=(299, 299, 3),
                                 output_size=28,
                                 warmup_epochs=2,
                                 regular_epochs=10,
                                 batch_size=16,
                                 checkpoint_path='./checkpoint/InceptionV3.h5',
                                 path_to_train="../dataset",
                                 image_size=299)
    model.train_inception_model()
