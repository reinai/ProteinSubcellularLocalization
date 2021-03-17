import pandas as pd

NUMBER_OF_FILTERS = 4

train_set_info = pd.read_csv("../dataset/train.csv")
print("1.) Inputs and outputs: \n" + str(train_set_info.head()))

print("\n2.) Total number of different images: " + str(train_set_info.shape[0]))
print("\n3.) Total number of images: " + str(train_set_info.shape[0] * NUMBER_OF_FILTERS))
