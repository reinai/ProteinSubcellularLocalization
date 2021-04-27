"""
Main idea lies in the fact that some protein patterns will more likely occur together. Correlation matrix is simplest
form of calculating the correlations, so we can use BMMs (Bernoulli Mixture Models).

author: Nikola Zubic
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import seaborn as sns
from label_information import fill_targets, label_names
from sklearn.model_selection import train_test_split
from BernoulliMixtureModels import BernoulliMixtureModel

# Set seaborn default interface
sns.set()

DATA_DIRECTORY = "../dataset"


train_set_labels = pd.read_csv(DATA_DIRECTORY + "/" + "train.csv")

for key in label_names.keys():
    train_set_labels[label_names[key]] = 0

train_set_labels = train_set_labels.apply(fill_targets, axis=1)

print(train_set_labels.head())

targets = train_set_labels.drop(["Id", "Target"], axis=1)

X = targets.values

x_train, x_test = train_test_split(X, shuffle=True, random_state=0)
components_to_test = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

scores = []
score = None
old_score = None

for n in range(len(components_to_test)):
    if n > 0:
        old_score = score
    model = BernoulliMixtureModel(components_to_test[n], 200)
    model.fit(x_train)
    score = model.score(x_test)
    scores.append(score)
    if n > 0:
        if score < old_score:
            estimated_components = components_to_test[n-1]
            break

model = BernoulliMixtureModel(number_of_components=25, maximum_number_of_iterations=200)
model.fit(X)

results = targets.copy()
results["cluster"] = np.argmax(model.gamma, axis=1)

print("\n1. How are specific protein patterns (organelles) distributed by percentage over clusters?")
print("""Several clusters have only one specific target protein.

97% of Mitochondria target proteins are located in cluster 18. Only a few percents are hold by cluster 12 and 8.

A lot of cellular components are hold by their own clusters (Golgi apparatus, Microtubules etc.).

Focal adhesion sites come together with Actin filaments and are spread over 3 clusters that are very similar up to 
absence or presence of plasma membrane, nuclear membrane and cell junctions.

Aggresomes come alone or sometimes with Cytoplasmic bodies. This makes sense: If bodies consist of viral capsids then 
there could be a lot of garbage or clutted proteins as well either due to immune response or to the viral building 
process.

Some components are spread over several clusters like nucleoplasm.
""")
grouped_targets = results.groupby("cluster").sum() / results.drop("cluster", axis=1).sum(axis=0) * 100
grouped_targets = grouped_targets.apply(np.round).astype(np.int32)

plt.figure(figsize=(20, 15))
sns.heatmap(grouped_targets, cmap="Purples", annot=True, fmt="g", cbar=False)
plt.title("How are specific protein patterns (organelles) distributed by percentage over clusters?")
plt.show()

cluster_names = {
    0: "Actin filaments & Focal adhesion sites",
    1: "Aggresomes",
    2: "Microtubules, Mitotic spindle, Cytokinetic Bridge",
    3: "RodsRings, Microtubule ends, Nuclear bodies",
    4: "Some nuclear membranes",
    5: "various - RodsRings",
    6: "various - Mitotic spindle, Organizing center, Cytosol",
    7: "Nuclear bodies & Aggresomes",
    8: "Nuclear speckles",
    9: "Nuclear membrane & Actin filaments & Focal adhesion sites",
    10: "Endoplasmatic reticulum & Endosomes & Lysosomes",
    11: "Low dense 1",
    12: "Mitotic spindle, Organizing center",
    13: "Plasma membrane & various",
    14: "Nucleoli fibrillar center & Peroxisomes",
    15: "Low dense 2",
    16: "Nucleoli fibrillar center & Cytoplasmic bodies",
    17: "Nucleoli & Microtubule ends & Peroxisomes & Rods Rings",
    18: "Mitochondria & Lipid droplets & RodsRings & Nucleoli",
    19: "Low dense 3",
    20: "Golgi apparatus",
    21: "Intermediate filaments",
    22: "Centrosome",
    23: "Cytoplasmic bodies & Aggresomes",
    24: "Lipid droplets & Peroxisomes & Cell junctions"
}
