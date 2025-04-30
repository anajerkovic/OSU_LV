import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

for n in range(1, 7):
    img = Image.imread("imgs\\test_"+ str(n) +".jpg")

    plt.figure()
    plt.title("Originalna slika" + str(n))
    plt.imshow(img)
    plt.tight_layout()

    img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    unique = np.unique(img_array, axis = 0)
    print("Broj boja u slici", str(n), ":", len(unique))

    km = KMeans(n_clusters = 5, init = 'k-means++')
    km.fit(img_array)
    img_array_p = km.predict(img_array)

    for i in range(len(img_array_p)):
        img_array[i] = km.cluster_centers_[img_array_p[i]]*255

    if n == 4: #zbog glupog New Yorka
        img_array = np.reshape(img_array, (w, h, d))
    else:
        img_array = np.reshape(img_array.astype(np.uint8), (w, h, d))

    plt.figure()
    plt.title("Obradjena slika" + str(n))
    plt.imshow(img_array)
    plt.tight_layout()

    img = Image.imread("imgs\\test_"+ str(n) +".jpg")
    img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    groups = np.arange(2, 11, 1)
    inertias = np.empty(len(groups))

    for nc in range(2, 11):
        km = KMeans(n_clusters = nc, init = 'k-means++')
        km.fit(img_array)
        inertias[nc-2] = km.inertia_

    plt.figure()
    plt.plot(groups, inertias)
    plt.title("J-K ovisnost")

    km = KMeans(n_clusters = 6, init = 'k-means++')
    km.fit(img_array)
    img_array_p = km.predict(img_array)

    for i in range(1, 7):
        img_array_k = np.full((w*h, d), 255)
        for j in range(len(img_array_p)):
            if img_array_p[j] == i-1:
                img_array_k[j] = km.cluster_centers_[i-1]*255
        img_array_k = np.reshape(img_array_k.astype(np.uint8), (w, h, d))
        plt.figure()
        plt.imshow(img_array_k)
    plt.show()
