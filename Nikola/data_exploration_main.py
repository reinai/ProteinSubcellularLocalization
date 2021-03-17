from data_exploration_functions import get_train_dataset, binary_target_values, organelle_count
from data_exploration_functions import number_of_organelles_per_image, correlation_matrix, ImageDataLoader
import matplotlib.pyplot as plt


if __name__ == "__main__":
    train_set_info = get_train_dataset()

    print("Dimensionality (rows/columns): " + str(train_set_info.shape[0]) + "/" +
          str(train_set_info.shape[1]) + "\n")

    correlation_matrix(train_set_info)

    train_set_info = train_set_info.apply(binary_target_values, axis=1)

    organelle_count(train_set_info)

    number_of_organelles_per_image(train_set_info)

    organelles_list = ["Lysosomes", "Endosomes",
                       "Microtubules", "Mitotic spindle", "Cytokinetic bridge",
                       "Actin filaments", "Focal adhesion sites",
                       "Rods & rings", "Microtubule ends", "Nuclear bodies"]

    ImageDataLoader.run(train_set_info, organelles_of_interest=organelles_list[0:2])
    ImageDataLoader.run(train_set_info, organelles_of_interest=organelles_list[2:5])
    ImageDataLoader.run(train_set_info, organelles_of_interest=organelles_list[5:7])
    ImageDataLoader.run(train_set_info, organelles_of_interest=organelles_list[7:10])
