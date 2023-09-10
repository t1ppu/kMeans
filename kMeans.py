# Importing required packages
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Reading the given file in the directory
root = "./UCI_datasets/"
file_name = sys.argv[1]
df = pd.read_csv(root+file_name, delim_whitespace=True)
# Divide into features and class label
X = df.iloc[:, :-1].values
y = df.iloc[:,1:].values
# Calculating SSE for 20 Iterations for different values of K
se = []
for k in range(2, 11):
 labels = np.random.randint(0, k, size=len(X))
 centroids = []
 for i in range(k):
 centroids.append(X[labels==i].mean(axis=0))
 se_k = []
 for i in range(20):
 kmeans = KMeans(n_init = 1, n_clusters=k, init=np.array(centroids))
 kmeans.fit(X)
 cluster_centers = kmeans.cluster_centers_
 cluster_labels = kmeans.labels_
 se_k.append(sum([np.linalg.norm(X[j]-cluster_centers[cluster_labels[j]])
for j in range(len(X))]))
 for j in range(k):
 centroids[j] = np.mean(X[cluster_labels == j], axis=0)
 # print(se_k)
 se.append(se_k[-1])
 print("For K=%d after 20 iterations: SSE error=%.4f" % (k, se[-1]))
# plot the SSE values vs K values
plt.plot(range(2, 11), se, 'bx-')
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('K-Means')
plt.show()