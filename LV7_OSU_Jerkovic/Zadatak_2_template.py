import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

images = ["test_1", "test_2", "test_3", "test_4", "test_5", "test_6"]



for image in images:
    img = Image.imread(f"imgs\\{image}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()


    km = KMeans(n_clusters=2, init="k-means++",n_init=5,random_state=0)
    km.fit(img_array_aprox)


    labels = km.predict(img_array_aprox)

    centers = km.cluster_centers_

    img_array_aprox[:,0] = centers[labels][:,0]
    img_array_aprox[:,1] = centers[labels][:,1]
    img_array_aprox[:,2] = centers[labels][:,2]

    img_array_aprox = np.reshape(img_array_aprox, (w,h,d))

    f,ax = plt.subplots(1,2)
    plt.title(f"Rezultat slike {image}")
    ax[0].imshow(img)
    ax[1].imshow(img_array_aprox)

    plt.tight_layout()
    plt.show()

inertias = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(img_array)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,10), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
