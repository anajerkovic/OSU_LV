import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


'''
Skripta zadatak_2.py ucitava podatkovni skup Palmer Penguins [1]. Ovaj 
podatkovni skup sadrži mjerenja provedena na tri razlicite vrste pingvina (Adelie, Chins- 
trap, Gentoo) na tri razlicita otoka u području Palmer Station, Antarktika. Vrsta pingvina 
odabrana je kao izlazna velicina i pri tome su klase označene s cjelobrojnim vrijednostima 
0, 1 i 2. Ulazne velicine su duljina kljuna (bill_length_mm) i duljina peraje u mm ( flipper_length_mm). 
Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna
funkcija plot_decision_region.
'''

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
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
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()[:,0]

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

'''
a) Pomocu stupčastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu 
pingvina) u skupu podataka za ucenje i skupu podataka za testiranje. Koristite numpy 
funkciju unique.
'''
classes, counts_train=np.unique(y_train, return_counts=True) # pronalayi sve jedinstvene vrijednosti klasa i broji koliko ih ima
classes, counts_test=np.unique(y_test, return_counts=True)

X_axis = np.arange(len(classes))
plt.bar(X_axis - 0.2, counts_train, 0.4, label = 'Train')
plt.bar(X_axis + 0.2, counts_test, 0.4, label = 'Test') 
plt.xticks(X_axis, ['Adelie(0)', 'Chinstrap(1)', 'Gentoo(2)'])
plt.xlabel("Penguins")
plt.ylabel("Counts")
plt.title("Number of each class of penguins, train and test data")
plt.legend()
plt.show()


'''
b) Izgradite model logisticke regresije pomo ˇ cu scikit-learn biblioteke na temelju skupa poda- ´
taka za ucenje.
'''
logRegModel = LogisticRegression(max_iter=120)
logRegModel.fit(X_train, y_train)

'''
c) Pronadite u atributima izgrađenog modela parametre modela. Koja je razlika u odnosu na 
binarni klasifikacijski problem iz prvog zadatka?
'''
theta0 = logRegModel.intercept_ #zbog broja klasa (3), ima 3 (za svaku klasu po jedan, jer je multinomial)(1xk dimenzije), kod binarne klasifikacije je bio 1 element(1 klasa)
coefs = logRegModel.coef_ 
print('Theta0:')
print(theta0)
print('Parametri modela') #zbog 3 klase, ima 3 retka parametara, svaki redak za jednu klasu, i 2 stupca, svaki stupac u paru s jednom ulaznom velicinom (kXm dimenzije)
print(coefs) #kod binarne klasifikacije je bio 1 red s 2 stupca (1 binarni klasifikator, 2 ulazne velicine)


'''
d) Pozovite funkciju plot_decision_region pri cemu joj predajte podatke za učenje i 
izgradeni model logističke regresije. Kako komentirate dobivene rezultate?
'''
plot_decision_regions(X_train, y_train, logRegModel)
plt.show()


'''
e) Provedite klasifikaciju skupa podataka za testiranje pomocu izgrađenog modela logisticke 
regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izračunajte tocnost.
Pomocu classification_report funkcije izracunajte vrijednost  cetiri glavne metrike na skupu podataka za testiranje
'''
y_test_p = logRegModel.predict(X_test)
cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
cm.plot()
plt.show()

print(f"Točnost: {accuracy_score(y_test,y_test_p)}")
print(classification_report(y_test,y_test_p))


'''
f) Dodajte u model još ulaznih velicina. Što se događa s rezultatima klasifikacije na skupu
podataka za testiranje?
'''
#dodavanjem parametra body mass, kvaliteta modela opada, dodavanjem bill_depth poraste, kao i dodavanjem oba parametra
