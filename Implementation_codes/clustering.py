import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from os import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None ,'display.max_columns', None)
import random as rn
np.random.seed(123)
rn.seed(123)

file_name = 'chiz_encoded_training_set'
df = pd.read_csv(file_name + '.dat', sep="\s+", header=None)

print("{0} rows and {1} columns".format(len(df.index), len(df.columns)))
print("")

X=df.iloc[:,[0,4,5,8,9,14,15,16,19,20,21,22,23,24,25,26,27,28,30,36,38,39,40,44,47]]

clusters=8

Y_Kmeans = KMeans(n_clusters = clusters, random_state=1)
Y_Kmeans.fit(X)
labels = Y_Kmeans.labels_
Y_Kmeans_silhouette = metrics.silhouette_score(X, labels, metric='sqeuclidean')
print("Silhouette for Kmeans: {0}".format(Y_Kmeans_silhouette))
print("Results for Kmeans: {0}".format(labels))

clusters = {}
n = 0
for item in labels:
	if item in clusters:
		clusters[item].append(df.iloc[[n]].to_string(index=False))
	else:
		clusters[item] = [df.iloc[[n]].to_string(index=False)]
	n +=1

for item in clusters:
	print("Cluster ", item)
	for i in clusters[item]:
		print(i)
