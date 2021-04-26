from inceptionnet.inception_nn_model import CustomInceptionModel

if __name__ == "__main__":
    """
    Main function to test our Inception model
    """

    model = CustomInceptionModel(input_shape=(299, 299, 3),
                                 output_size=28,
                                 warmup_epochs=2,
                                 regular_epochs=50,
                                 batch_size=16,
                                 checkpoint_path='./inceptionnet/checkpoint/InceptionV3.h5',
                                 path_to_train="./dataset",
                                 path_to_test="./dataset",
                                 image_size=299)
    print('Result -> ', model.calculate_predicted_macro_f1_score(
           predicted_csv_path='./dataset/predicted_InceptionV3.csv'))
