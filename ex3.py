import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans


I = Image.open("img1.jpg")
a = np.asarray(I, dtype=np.float32)/255    # chuan hoa ve 0->1
plt.imshow(a)
plt.axis('off')
plt.show()

w, h = I.size
colors = I.getcolors(w * h)
num_colors = len(colors)
num_pixels = w * h
print('Number of pixels = ', num_pixels)
print('Number of colors = ', num_colors)

x, y, z = a.shape
print('a shape ', a.shape)
a1 = a.reshape(x*y, z)
print('a1 shape ', a1.shape)

n = 100
k_means = KMeans(n_clusters=n)
k_means.fit(a1)
centroids = k_means.cluster_centers_
labels = k_means.labels_
print('centroids shape ', centroids.shape)
print('labels shape ', labels.shape)
a2 = centroids[labels]
print('a2 shape ', a2.shape)

a3 = a2.reshape(x,y,z)
image = plt.figure()

print('a3 shape ', a3.shape)
plt.imshow(a3)
plt.axis('off')
plt.savefig('n_'+str(n)+'.jpg')
plt.show()