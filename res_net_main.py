# author: Nikola Zubic

from gap_net_main import getDataset, macro_F1_score


def test_res_net():
    image_folder_path = input("\nEnter test image folder path: ")
    our_test_predictions_directory = input("\nEnter file path to our predictions: ")
    test_data_directory = input("\nEnter file path to the test data directory: ")

    paths_test, y_pred = getDataset(image_folder_path, predictions_csv_path=our_test_predictions_directory)
    _, y = getDataset(image_folder_path, predictions_csv_path=test_data_directory)

    print("Macro F1 score: " + str(macro_F1_score(y_true=y, y_predicted=y_pred)))


if __name__ == "__main__":
    test_res_net()

"""
/home/nikola/Documents/TEST_PROTEIN/test
/home/nikola/Desktop/ds_projekat/ProteinSubcellularLocalization/models_weights_predictions/res_net/predictions_res_net.csv
/home/nikola/Documents/TEST_PROTEIN/test.csv
"""
