import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

'''
Iris Dataset sastoji se od informacija o laticama i cašicama tri razlicita cvijeta 
irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
from sklearn import datasets
iris = datasets.load_iris()
Upoznajte se s datasetom i dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na 
sljedeca pitanja:
'''
iris = datasets.load_iris()

data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['Iris Type'] = iris.target # 0 -sentosa, 1 - versicolor, 2 - virginca

print(data.head(10))
data.info()
'''
a) Prikažite odnos duljine latice i cašice svih pripadnika klase Virginica pomocu scatter
dijagrama zelenom bojom. Dodajte na isti dijagram odnos duljine latice i cašice svih 
pripadnika klase Setosa, sivom bojom. Dodajte naziv dijagrama i nazive osi te legendu.
Komentirajte prikazani dijagram.
'''
virginica = data[(data['Iris Type'] == 2)]
versicolor = data[(data['Iris Type'] == 1)]
setosa = data[(data['Iris Type'] == 0)]

plt.scatter(virginica['sepal length (cm)'], virginica['petal length (cm)'], color='green', label='Virginica')
plt.scatter(setosa['sepal length (cm)'], setosa['petal length (cm)'], color='gray', label='Setosa')

plt.title('Odnos duljine čašice i latice za vrste Virginica i Setosa')
plt.xlabel('Duljina čašice (cm)')
plt.ylabel('Duljina latice (cm)')
plt.legend()
plt.show()

'''
b) Pomocu stupcastog dijagrama prikažite najvecu vrijednost širine cašice za sve tri klase 
cvijeta. Dodajte naziv dijagrama i nazive osi. Komentirajte prikazani dijagram.
'''
labels = ['Setosa', 'Versicolor', 'Virginica']
setosa_biggest = (setosa['sepal width (cm)']).max()
versicolor_biggest = (versicolor['sepal width (cm)']).max()
virginica_biggest = (virginica['sepal width (cm)']).max()

sizes = [setosa_biggest, versicolor_biggest, virginica_biggest]
colors = ['magenta', 'purple', 'yellow']

plt.bar(labels,sizes,color = colors)
plt.title("Najveće vrijednosti širine čašica")
plt.xlabel("Klasa cvijeta")
plt.ylabel("Veličnia čašice")
plt.show()

# c) Koliko jedinki pripadnika klase Setosa ima vecu širinu cašice od prosjecne širine cašice te klase?
setosa_mean = (setosa['sepal width (cm)']).mean()
print(f"Broj jedinki koje imaju veću širinu čašice od prosjeka: {len(setosa[(setosa['sepal width (cm)'] > setosa_mean)])}")

'''
Zadatak 0.0.8 Iris Dataset sastoji se od informacija o laticama i cašicama tri razlicita cvijeta irisa (Setosa, Versicolour i Virginica). 
Dostupan je u sklopu bibilioteke scikitlearn. Upoznajte se s datasetom. 
Pripremite podatke za ucenje. 
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja:
'''
input_var = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = data[input_var]

scaler = StandardScaler()
X_s = scaler.fit_transform(X)



# a) Pronadite optimalni broj klastera K za klasifikaciju cvijeta irisa algoritmom K srednjih vrijednosti.
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_s)
    inertia.append(kmeans.inertia_)

# b) Graficki prikažite lakat metodu.
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow metoda za pronalaženje optimalnog K')
plt.xlabel('Broj klastera (K)')
plt.ylabel('Inercija')
plt.show()

'''
Primijenite algoritam K srednjih vrijednosti koji ce pronaci grupe u podatcima. 
Koristite vrijednot K dobivenu u prethodnom zadatku.
'''
kmeans = KMeans(n_clusters=3, random_state=42) # iz grafa vidimo da je najbolji K = 3
kmeans.fit(X_s)
labels = kmeans.predict(X_s)

'''
d) Dijagramom raspršenja prikažite dobivene klastere. Obojite ih razlicitim bojama (zelena, žuta i narancasta). 
Centroide obojite crvenom bojom. 
Dodajte nazive osi, naziv dijagrama i legendu. 
Komentirajte prikazani dijagram.
'''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_s)
centroids_pca = pca.transform(kmeans.cluster_centers_)

cluster_colors = ['green', 'yellow', 'orange']
# Crtanje klastera
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1],
                c=cluster_colors[i], label=f'Klaster {i}', alpha=0.6)

# Crtanje centroida
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='red', marker='X', label='Centroidi')

# Oznake i legenda
plt.xlabel('PCA komponenta 1')
plt.ylabel('PCA komponenta 2')
plt.title('K-means klasteriranje Iris podataka (K=3)')
plt.legend()
plt.grid(True)
plt.show()

# e) Usporedite dobivene klase sa njihovim stvarnim vrijednostima. Izracunajte tocnost klasifikacije.
y_true = data['Iris Type'].to_numpy()
y_pred = labels

disp = ConfusionMatrixDisplay( confusion_matrix ( y_true , y_pred ) )
disp.plot()
plt.show()