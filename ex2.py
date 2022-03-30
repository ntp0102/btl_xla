import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


digits = load_digits()
data = digits.data
print(data[0])
print(data[0].shape)
print(data.shape)
# exit()
data = 255-data         # convert nen den->trang, mau chu trang->den
np.random.seed(1)
n = 10
kmeans = KMeans(n_clusters=n, init='random')
kmeans.fit(data)
Z = kmeans.predict(data)
for i in range(0,n):
    row = np.where(Z==i)[0]         # row in Z for elements of cluster i
    num = row.shape[0]              # number of elements for each cluster
    r = int(np.floor(num/10.))      # number of rows in the figure of the cluster
    print("cluster "+str(i))
    print(str(num)+" elements")

    # plt.figure(figsize=(10,10))
    for k in range(0, num):
        plt.subplot(r+1, 10, k+1)
        image = data[row[k], ]
        image = image.reshape(8, 8)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    # plt.title("Cluster" + str(i), loc='right')
    plt.show()