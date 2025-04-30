import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score



'''
Skripta zadatak_1.py ucitava Social_Network_Ads.csv skup podataka .
Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
Podaci o korisnicima su spol, dob i procijenjena placa. 
Razmatra se binarni klasifikacijski problem gdje su dob i procijenjena placa ulazne veličine, dok je kupovina (0 ili 1) izlazna velicina. 
Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija plot_decision_region [1].
Podaci su podijeljeni na skup za ucenje i skup za testiranje modela u omjeru 80%-20% te su standardizirani. 
Izgraden je model logističke regresije te je izračunata njegova tocnost na skupu podataka za učenje i skupu podataka za testiranje.
'''
def plot_decision_regions(X, y, classifier, resolution=0.02):
    
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


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy() # ulazne velicine
y = data["Purchased"].to_numpy() # izlazne velicine

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
print()

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost modela logističke regresije: \n " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


'''
Izradite algoritam KNN na skupu podataka za ucenje (uz  K=5). 
Izracunajte točnost klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje.
Usporedite dobivene rezultate s rezultatima logisticke regresije. 
Što primjećujete vezano uz dobivenu granicu odluke KNN modela?
'''

KNN_model = KNeighborsClassifier( n_neighbors = 5 )
KNN_model.fit( X_train_n , y_train)

y_test_p_KNN = KNN_model.predict(X_test_n)
y_train_p_KNN = KNN_model.predict(X_train_n)

print("KNN model:")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))
print()

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()
'''
U usporedbi sa rezultatima logističke regresije, KNN ima veću točnost te bolje klasificira zadane podatke, granica odluke
nije linearna i na ovom primjeru bolje opisuje podatke. 
'''
KNN_1 = KNeighborsClassifier( n_neighbors = 1 )
KNN_1.fit( X_train_n , y_train)

y_test_p_KNN_1 = KNN_1.predict(X_test_n)
y_train_p_KNN_1 = KNN_1.predict(X_train_n)

print("KNN model sa K=1:")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN_1))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN_1))))
plt.subplot(1,2,1)
plot_decision_regions(X_train_n, y_train, classifier=KNN_1)

plt.title("K = 1")

KNN_100 = KNeighborsClassifier( n_neighbors = 100 )
KNN_100.fit( X_train_n , y_train)

y_test_p_KNN_100 = KNN_100.predict(X_test_n)
y_train_p_KNN_100 = KNN_100.predict(X_train_n)

print("KNN model sa K=100:")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN_100))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN_100))))
plt.subplot(1,2,2)
plot_decision_regions(X_train_n, y_train, classifier=KNN_100)
plt.title("K = 100")
plt.show()
print()
'''
Kako izgleda granica odluke kada je K =1 i kada je K = 100?
Kada je K = 1 granica je oko svakog pojedinačnog podatak - overfitting, a kada je K = 100 nije dovoljno dobra raspodjela - underfitting
'''



'''
Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma KNN za podatke iz Zadatka 1.
'''
k_values = range(1, 20)
cv_scores = []
 # prolazim kroz k-ove, za svaki radim model i bilježim srednju točnost
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_n, y_train, cv=5)  # koristi standardizirane podatke
    cv_scores.append(scores.mean())

# pronađi najbolji K
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimalna vrijednost K: {optimal_k}")
print(f"Najbolja srednja točnost: {max(cv_scores):.3f}")
print()
# crtanje
plt.plot(k_values, cv_scores, marker='o', c='r')
plt.xlabel('K vrijednost')
plt.ylabel('Srednja točnost')
plt.title('Odabir optimalnog K za KNN')
plt.show()


'''
Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju te prikažite dobivenu granicu odluke. 
Mijenjajte vrijednost hiperparametra C i gamma. Kako promjena ovih hiperparametara utjece na granicu odluke te pogrešku na skupu podataka za testiranje? 
Mijenjajte tip kernela koji se koristi. Što primjecujete?
'''
SVM_model = svm.SVC( kernel ='rbf', gamma =0.01 , C=10 )
SVM_model.fit( X_train_n , y_train )
y_test_p_SVM = SVM_model.predict(X_test)
y_train_p_SVM = SVM_model.predict(X_train_n)

print("SVM:")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))

plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost SVM modela: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

'''
def SVM_different_c_and_gamma(C,gamma,plot):
    SVM = svm.SVC( kernel ='rbf', gamma = gamma , C=C )
    SVM.fit(X_train_n, y_train)
    plt.subplot(1,3,plot)
    plot_decision_regions(X_train_n, y_train, classifier=SVM)
    plt.title(f"SVM with gamma = {gamma} and C = {C}")

SVM_different_c_and_gamma(1,1,1)
SVM_different_c_and_gamma(0.1,0.1,2)
SVM_different_c_and_gamma(10,1,3)
plt.show()
'''
'''
Povećanjem hiperparametra C, granica odluke postaje složenija jer model pokušava savršeno razdvojiti podatke, što može dovesti do overfittinga. 
Povećanjem gamma, model postaje osjetljiviji na pojedinačne primjere, što također može uzrokovati pretjerano prilagođavanje i 
lošiju generalizaciju na testnom skupu.
'''

'''
Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i gamma algoritma SVM za problem iz Zadatka 1.
'''
param_grid = {'C': [0.001, 0.01, 0.1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
svm_gscv = GridSearchCV( SVM_model , param_grid , cv =5 , scoring ='accuracy', n_jobs = -1 )
svm_gscv.fit ( X_train , y_train )
print ( svm_gscv.best_params_ )
print(svm_gscv.best_score_)

# najbolji - C:10, gamma: 0.01