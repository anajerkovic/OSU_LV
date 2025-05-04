import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
'''
Zadatak 0.0.4 Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo 1912. godine. 
Upoznajte se s datasetom i dodajte programski kod u skriptu pomocu kojeg možete 
odgovoriti na sljedeca pitanja:
'''
data = pd.read_csv('titanic.csv')

print(data.head(15))
# a) Za koliko žena postoje podatci u ovom skupu podataka?
print(f"Broj žena na titaniku: {len(data[(data['Sex'] == "female")])}")

# b) Koliki postotak osoba nije preživio potonuce broda?
not_survived = data[(data['Survived'] == 0)]
survived = data[(data['Survived'] == 1)]
print(f"Postotak nepreživjelih na titaniku: {(len(not_survived))/(len(data))*100:.2f}%")
print(f"Postotak preživjelih na titaniku: {(len(survived))/(len(data))*100:.2f}%")

'''
c) Pomocu stupcastog dijagrama prikažite postotke preživjelih muškaraca (zelena boja) i žena (žuta boja). 
Dodajte nazive osi i naziv dijagrama. 
Komentirajte korelaciju spola i postotka preživljavanja.
'''
# Ukupni broj muškaraca i žena
total_men = data[data['Sex'] == 'male']
total_women = data[data['Sex'] == 'female']
# Broj preživjelih muškaraca i žena
survived_men = survived[survived['Sex'] == 'male']
survived_women = survived[survived['Sex'] == 'female']
# Postotci
percent_men = (len(survived_men) / len(total_men)) * 100
percent_women = (len(survived_women) / len(total_women)) * 100

labels = ['Muškarci', 'Žene']
percentages = [percent_men, percent_women]
colors = ['green', 'yellow']

# Stupčasti dijagram
plt.bar(labels, percentages, color=colors)
plt.title('Postotak preživjelih muškaraca i žena')
plt.ylabel('Postotak (%)')
plt.xlabel('Spol')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.8)
plt.show()

# d) Kolika je prosjecna dob svih preživjelih žena, a kolika je prosjecna dob svih preživjelih muškaraca?
print(f"Projescna dob svih preživjelih žena: {survived_women['Age'].mean():.2f} godina")
print(f"Projescna dob svih preživjelih muškaraca: {survived_men['Age'].mean():.2f} godina")

# e) Koliko godina ima najstariji preživjeli muškarac u svakoj od klasa? Komentirajte.
biggest_age = survived_men.groupby('Pclass')['Age'].max()
print(biggest_age)


'''
Zadatak 0.0.5 Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo 1912. godine. 
Upoznajte se s datasetom. Ucitajte dane podatke. 
Podijelite ih na ulazne podatke X predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem Survived. 
Podijelite podatke na skup za ucenje i skup za testiranje modela u omjeru 70:30.
Izbacite izostale i null vrijednosti. 
Skalirajte podatke. 
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja:
'''
# podjela na ulazne i izlazne podatke i ciscenje null vrijednosti
input_var = ['Pclass', 'Sex', 'Fare', 'Embarked']
data = data.dropna()
X = data[input_var]
y = data['Survived']

# razdvajanje kategorickih i numerickih ulaznih vrijednosti - zbog skaliranja
X_numerical=X[['Pclass','Fare']]
X_categorical=X[['Sex', 'Embarked']]

# skaliranje numerickih varijabli
sc = StandardScaler()
X_numerical_n = sc.fit_transform(X_numerical)
X_numerical_n=pd.DataFrame({'Pclass':X_numerical_n[:,0], 'Fare':X_numerical_n[:,1]})
#skaliranje kategorickih varijabli
encoder = OneHotEncoder(sparse_output=False)
X_categorical_n = encoder.fit_transform(X_categorical)
X_categorical_n=pd.DataFrame({'Sex':X_categorical_n[:,0], 'Embarked':X_categorical_n[:,1]})
print(X_categorical_n)
#spajanje obje varijable nazad u jednu ulaznu X varijablu
X_n=pd.concat([X_numerical_n,X_categorical_n],axis=1)
#razdvajanje na skup za ucenje i skup za testiranje
X_train_n, X_test_n, y_train, y_test = train_test_split(X_n, y, test_size = 0.3, stratify=y, random_state = 10)

# a) Izradite algoritam KNN na skupu podataka za ucenje (uz K=5). 
# Vizualizirajte podatkovne primjere i granicu odluke.
KNN_model = KNeighborsClassifier( n_neighbors = 5 )
KNN_model.fit( X_train_n , y_train)
y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)

# vizualizacija podataka - ne znam kako napraviti
'''
def plot_decision_regions(X, y, classifier, resolution=0.02):
    X = X.values  # Konvertiraj u NumPy array ako je DataFrame
    y = y.values if isinstance(y, pd.Series) else y 


    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)
plot_decision_regions(X_train_n, y_train, KNN_model)
'''

# b) Izracunajte tocnost klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje. Komentirajte dobivene rezultate
print("KNN model sa K = 5: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
print()

# c) Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma KNN.


# Definiranje raspona vrijednosti K koje želimo isprobati
param_grid = {'n_neighbors': np.arange(1, 100)}
# KNN model
knn = KNeighborsClassifier()
# GridSearch s 5-strukom unakrsnom validacijom (5-fold cross-validation)
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_n, y_train)
# Najbolji rezultat
print(f"Najbolji K: {grid_search.best_params_['n_neighbors']}")
print(f"Najbolja točnost: {grid_search.best_score_:.4f}")


KNN_26 = KNeighborsClassifier(n_neighbors=26)
KNN_26.fit( X_train_n , y_train)
y_train_p_26 = KNN_26.predict(X_train_n)
y_test_p_26 = KNN_26.predict(X_test_n)

'''
d) Izracunajte to cnost klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje 
za dobiveni K. 
Usporedite dobivene rezultate s rezultatima kada je K=5.
'''
print("KNN model sa K = 26: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_26))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_26))))
print()



'''
Zadatak 0.0.6 Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo 1912. godine. 
Upoznajte se s datasetom. Ucitajte dane podatke. 
Podijelite ih na ulazne podatke  X predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem Survived. 
Podijelite podatke na skup za ucenje i skup za testiranje modela u omjeru 75:25. 
Izbacite izostale i null vrijednosti. Skalirajte podatke. 
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja:
'''
'''
a) Izgradite neuronsku mrežu sa sljedecim karakteristikama: 
- model ocekuje ulazne podatke X
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
- treci skriveni sloj ima 4 neurona i koristi relu aktivacijsku funkciju
- izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
Ispišite informacije o mreži u terminal.
'''
print('Train: X=%s, y=%s' % (X_train_n.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test_n.shape, y_test.shape))

# OVAJ DIO NECE RADITI JER MI NE RADI TENSORFLOW TAKO JE NADALJE SAMO WISHFUL THINKING
model = keras.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(X_train_n.shape[1],)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary() # informacije o mrezi


'''
b) Podesite proces treniranja mreže sa sljedecim parametrima:
- loss argument: binary_crossentropy
- optimizer: adam
- metrika: accuracy
'''
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# c) Pokrenite ucenje mreže sa proizvoljnim brojem epoha (pokušajte sa 100) i velicinom batch-a 5.
model.fit(X_train_n, y_train, batch_size=5, epochs=100, validation_split=0.1)

# d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju ucitanog modela.
model.save("model.keras")

# e) Izvršite evaluaciju mreže na testnom skupu podataka
from keras import models
model = models.load_model('model.keras') # ucitavanje spremljenog modela
loss, accuracy = model.evaluate(X_test_n, y_test)
print(f"Testni gubitak (loss): {loss}")
print(f"Testna točnost (accuracy): {accuracy}")


'''
f) Izvršite predikciju mreže na skupu podataka za testiranje. 
Prikažite matricu zabune za skup podataka za testiranje. 
Komentirajte dobivene rezultate i predložite kako biste ih poboljšali, ako je potrebno.
'''
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Matrica zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matrica zabune za testni skup")
plt.show()