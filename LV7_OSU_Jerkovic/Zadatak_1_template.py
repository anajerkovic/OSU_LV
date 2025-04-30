import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering

'''
Skripta zadatak_1.py sadrži funkciju generate_data koja služi za generiranje
umjetnih podatkovnih primjera kako bi se demonstriralo grupiranje. Funkcija prima cijeli broj
koji definira željeni broju uzoraka u skupu i cijeli broj od 1 do 5 koji definira na koji nacin ce
se generirati podaci, a vraca generirani skup podataka u obliku numpy polja pri cemu su prvi i 
drugi stupac vrijednosti prve odnosno druge ulazne velicine za svaki podatak. Skripta generira 
500 podatkovnih primjera i prikazuje ih u obliku dijagrama raspršenja.
'''

def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # 2 grupe
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)

    else:
        X = []

    return X




groups = np.arange(2, 11, 1)
inertias = np.empty(len(groups))
best_nc = [3, 6, 4, 6, 6]

for n in range(1, 6):
    X = generate_data(500, n)
    plt.figure()
    plt.scatter(X[:,0],X[:,1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('podatkovni primjeri (flagc = '+str(n)+')')

    for nc in range(2, 7):
        km = KMeans(n_clusters = nc, init = 'k-means++')
        km.fit(X)
        X_p = km.predict(X)
        plt.figure()
        plt.scatter(x = X[:,0], y = X[:,1], c = X_p, cmap = 'Set1')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('grupirani primjeri (n_c = '+str(nc)+')')

    for nc in range(2, 11):
        km = KMeans(n_clusters = nc, init = 'k-means++')
        km.fit(X)
        inertias[nc-2] = km.inertia_
    plt.figure()
    plt.plot(groups, inertias)
    plt.title('J-K ovisnost (flagc = '+str(n)+')')

    km = KMeans(n_clusters = best_nc[n-1], init = 'k-means++')
    km.fit(X)
    X_p = km.predict(X)
    plt.figure()
    plt.scatter(x = X[:,0], y = X[:,1], c = X_p, cmap = 'Set1')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('flagc = '+str(n)+', optimalni K = '+str(best_nc[n-1]))
    plt.show()
