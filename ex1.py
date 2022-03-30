import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
np.random.seed(4)

x1 = np.random.standard_normal((100, 2))*0.7 + np.ones((100, 2))
x2 = np.random.standard_normal((100, 2))*0.5 - 0.8*np.ones((100, 2))
x3 = np.random.standard_normal((100, 2)) - 2*np.ones((100, 2)) + 4
X = np.concatenate((x1, x2, x3), axis=0)

n = 5
k_means = KMeans(n_clusters=n)
k_means.fit(X)
centroids = k_means.cluster_centers_
labels = k_means.labels_

# plt.plot(X[labels==0, 0], X[labels==0, 1], '.' , label='initial')

plt.plot(X[labels==0,0],X[labels==0,1],'r.', label='cluster 1')
plt.plot(X[labels==1,0],X[labels==1,1],'b.', label='cluster 2')
plt.plot(X[labels==2,0],X[labels==2,1],'g.', label='cluster 3')
plt.plot(X[labels==3,0],X[labels==3,1],'y.', label='cluster 4')
plt.plot(X[labels==4,0],X[labels==4,1],'.', label='cluster 5')
plt.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')

plt.show()
