# author: Nikola Zubic
import numpy as np
import pandas as pd
from label_information import fill_targets, label_names
import seaborn as sns
from sklearn.model_selection import train_test_split
from BernoulliMixtureModels import BernoulliMixtureModel
import matplotlib.pyplot as plt

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
print("\n1. Which clusters have anomalies?")
print("""If we compare this threshold with the distribution of the sample log-likelihoods, we can see that below this 
threshold there are some samples in the negative side that are outliers compared to the other ones.
""")

sample_logLs = model.score_samples(X)

"""
For each sample we can try to find out how dense the region of the target feature space is, hence how many other samples 
are around it with the same kind of target structure. This density is given by the sample log-likelihood.
"""
my_threshold = np.quantile(sample_logLs, 0.05)

plt.figure(figsize=(20, 5))
sns.distplot(sample_logLs)
plt.axvline(my_threshold, color="Orange")
plt.xlabel("Sample log likelihood of Bernoulli's mixture")
plt.title("Which clusters have anomalies?")
plt.ylabel("Density")
plt.show()

########################################################################################################################
print("\n2. Anomaly percentage by clusters")
results["cluster_names"] = results.cluster.apply(lambda l: cluster_names[l])
cluster_ids = np.arange(0, 25)
names = [cluster_names[l] for l in cluster_ids]
results["anomaly"] = np.where(sample_logLs <= my_threshold, 1, 0)

anomalies = results.groupby("cluster_names").anomaly.value_counts() / results.groupby("cluster_names").cluster.count() \
            * 100
anomalies = anomalies.unstack()
anomalies.fillna(0, inplace=True)
anomalies = anomalies.apply(np.round)
anomalies = anomalies.astype(np.int)

plt.figure(figsize=(20, 5))
sns.heatmap(anomalies.transpose(), cmap="YlOrBr", annot=True, square=True, cbar=False)
plt.title("Anomaly percentage by clusters")
plt.show()

########################################################################################################################
print("\n3. Cluster interactions")
print("""Aggresomes are often visible with the Mitotic spindle & Organizing center cluster.

The various Rods & Rings cluster is often seen with the Nucleoli fibrillar center & Cytoplasmic bodies cluster.
""")

results["alternative_cluster"] = np.argsort(model.gamma, axis=1)[:, -2]
results["alternative_names"] = results.alternative_cluster.apply(lambda l: cluster_names[l])
results["alternative_certainties"] = np.sort(model.gamma, axis=1)[:, -2]

competition = np.round(100 * results.groupby(
    "cluster_names").alternative_names.value_counts() / results.groupby(
    "cluster_names").alternative_names.count())
competition = competition.unstack()
competition.fillna(0, inplace=True)

plt.figure(figsize=(20, 15))
sns.heatmap(competition, cmap="winter", annot=True, fmt="g", square=True, cbar=False)
plt.title("Cluster interactions")
plt.show()
