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

########################################################################################################################
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

########################################################################################################################
print("\n2. In one certain cluster what is the percentage of presence of a certain organelle?")
print("""This shows us what kind of protein patterns (organelles) we will always find per cluster. For example, 
endoplasmatic reticulum, lysosome and endosome cluster shows that almost all samples hold the endoplasmatic reticulum 
with 1. In contrast, due to the rareness and omnipresence of lysosomes and endosomes the hotness of them is very low. A 
lot of samples do not have them even though all targets of lysosomes and endosomes can be find within that cluster.

Gives us more insights of the imbalance of target proteins per cluster.

Nucleoplasm and Cytosol are the dominating targets that influence the composition of many clusters.

Rods and Rings and Microtubule ends are dominated by more common classes like Nucleoplasm, Nucleoli and Nuclear bodies.

Once we have found the cluster 1 we can definitely say that all of its samples have an Aggresome.
""")
cluster_size = results.groupby("cluster").Nucleoplasmn.count()
cluster_composition = results.groupby("cluster").sum().apply(lambda l: l / cluster_size, axis=0) * 100
cluster_composition = cluster_composition.apply(np.round).astype(np.int)

cluster_composition = cluster_composition.reset_index()
cluster_composition.cluster = cluster_composition.cluster.apply(lambda l: cluster_names[l])
cluster_composition = cluster_composition.set_index("cluster")

plt.figure(figsize=(20, 20))
sns.heatmap(cluster_composition, cmap="PuBuGn", annot=True, fmt="g", cbar=False)
plt.title("In one certain cluster what is the percentage of presence of a certain organelle?")
plt.ylabel("")
plt.show()

########################################################################################################################
print("\n3. Which clusters are more probable to appear?")
results["cluster_names"] = results.cluster.apply(lambda l: cluster_names[l])
cluster_ids = np.arange(0, 25)
names = [cluster_names[l] for l in cluster_ids]

pi = pd.Series(data=model.pi, index=names).sort_values(ascending=False)
plt.figure(figsize=(20, 5))
sns.barplot(x=pi.index, y=pi.values, palette="Set2", order=pi.index)
plt.xticks(rotation=90)
plt.title("Which clusters are more probable to appear?")
plt.show()

########################################################################################################################
print("\n4. What is the number of samples per clusters?")
cluster_counts = results.groupby("cluster").cluster.count()
cluster_counts = cluster_counts.sort_values()
names = [cluster_names[num] for num in cluster_counts.index]

plt.figure(figsize=(20, 5))
sns.barplot(x=names, y=cluster_counts.values, order=names, palette="ch:s=.25,rot=-.25")
plt.xticks(rotation=90)
plt.title("What is the number of samples per clusters?")
plt.show()

########################################################################################################################
print("\n5. What is the number of targets per clusters?")
print("""Low dense clusters have multiple targets.

Many samples only have one or two target proteins.

There are only two clusters that have at least two targets present: Nucleoli fibrillar center & Cytoplasmic bodies and
Rods & Rings, Microtubule ends, Nuclear bodies.
""")

results["number_of_targets"] = results.drop("cluster", axis=1).sum(axis=1)

multilabel_stats = results.groupby("cluster_names").number_of_targets.value_counts()
multilabel_stats /= results.groupby("cluster_names").number_of_targets.count()
multilabel_stats = multilabel_stats.unstack()
multilabel_stats.fillna(0, inplace=True)
multilabel_stats = 100 * multilabel_stats
multilabel_stats = multilabel_stats.apply(np.round)
multilabel_stats = multilabel_stats.astype(np.int)

plt.figure(figsize=(20, 5))
plt.title("5. What is the number of targets per clusters?")
sns.heatmap(multilabel_stats.transpose(),
            square=True,
            cbar=False,
            cmap="Reds",
            annot=True)
plt.show()

########################################################################################################################
