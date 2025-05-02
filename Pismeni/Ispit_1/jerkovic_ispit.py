import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay, accuracy_score, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


'''
Zadatak 0.0.1 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu otkrivanja dijabetesa, pri cemu se u 
devetom stupcu nalazi klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes). 
Ucitajte dane podatke u obliku numpy polja data. 
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeća pitanja: 
'''
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

data = pd.read_csv('pima-indians-diabetes.csv', skiprows=9, header=None, names=column_names)
data.info()
print(data.head(10))

# a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja?
print(f"Mjerenja su izvršena na {data.shape[0]} osoba")

#b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne
# mase (BMI)? Obrišite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?
# Provjera koliko ima NaN i nula u stupcima 'Age' i 'BMI'
print("Izostale vrijednosti (NaN):")
print(data[['Age', 'BMI']].isnull().sum())

print("\nBroj nula u 'Age' i 'BMI':")
print((data[['Age', 'BMI']] == 0).sum())

# Uklanjanje redaka koji imaju NaN ili 0 u 'Age' ili 'BMI'
data_cleaned = data.dropna(subset=['Age', 'BMI'])  # prvo makni NaN
data_cleaned = data_cleaned[(data_cleaned['Age'] != 0) & (data_cleaned['BMI'] != 0)]  # makni 0

# Rezultat
print(f"\nBroj preostalih uzoraka nakon čišćenja: {len(data_cleaned)}")


'''
c) Prikažite odnos dobi i indeksa tjelesne mase (BMI) osobe pomocu scatter dijagrama.
Dodajte naziv dijagrama i nazive osi s pripadajucim mjernim jedinicama. 
Komentirajte odnos dobi i BMI prikazan dijagramom
'''
plt.scatter(data_cleaned['Age'], data_cleaned['BMI'], c="magenta")
plt.title("Odnos dobi i indeksa tjelesne mase(BMI)")
plt.xlabel("Age (years)")
plt.ylabel("BMI (weight in kg/(height in m)^2)")
plt.show()

'''
d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne 
mase (BMI) u ovom podatkovnom skupu.
'''
print(f"Minimalna vrijednost BMI: {data_cleaned['BMI'].min()}")
print(f"Maksimalna vrijednost BMI: {data_cleaned['BMI'].max()}")
print(f"Srednja vrijednost BMI: {data_cleaned['BMI'].mean()}")
print()
'''
e) Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one
kojima nije. 
Kolikom je broju ljudi dijagonosticiran dijabetes? 
Komentirajte dobivene vrijednosti.
'''
diabetes = data_cleaned[(data_cleaned['Outcome'] == 1)]
no_diabetes = data_cleaned[(data_cleaned['Outcome'] == 0)]
print("Osobe sa dijabetesom:")
print(f"Minimalna vrijednost BMI: {diabetes['BMI'].min()}")
print(f"Maksimalna vrijednost BMI: {diabetes['BMI'].max()}")
print(f"Srednja vrijednost BMI: {diabetes['BMI'].mean()}")
print()

print("Osobe koje nemaju dijabetes:")
print(f"Minimalna vrijednost BMI: {no_diabetes['BMI'].min()}")
print(f"Maksimalna vrijednost BMI: {no_diabetes['BMI'].max()}")
print(f"Srednja vrijednost BMI: {no_diabetes['BMI'].mean()}")
print()

print(f"Dijabetes je dijagnosticiran {diabetes.shape[0]} osoba.")




'''
Zadatak 0.0.2 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
otkrivanja dijabetesa, pri cemu se u devetom stupcu nalazi izlazna veličina, predstavljena klasom 0 (nema dijabetes) ili klasom 1 (ima dijabetes).
Ucitajte dane podatke u obliku numpy polja data. 
Podijelite ih na ulazne podatke X i izlazne podatke y.
Podijelite podatke na skup za ucenje i skup za testiranje modela u omjeru 80:20.
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja:
'''

input_var = ['Pregnancies',
             'Glucose',
             'BloodPressure',
             'SkinThickness',
             'Insulin',
             'BMI',
             'DiabetesPedigreeFunction',
             'Age']

X = data_cleaned[input_var]
y = data_cleaned['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =1 )


# a) Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa podataka za ucenje.
logRegModel = LogisticRegression(max_iter=1200)
logRegModel.fit(X_train, y_train)


# b) Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke regresije.
# c) Izracunajte i prikažite matricu zabune na testnim podacima. Komentirajte dobivene rezultate.
# d) Izracunajte tocnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte dobivene rezultate.
y_test_p = logRegModel.predict( X_test )
print("=== Evaluacija modela ===")
print(f"Točnost (accuracy): {accuracy_score(y_test, y_test_p)}")
print(f"Preciznost (precision): {precision_score(y_test, y_test_p)}")
print(f"Odziv (recall): {recall_score(y_test, y_test_p)}")
print("\nKlasifikacijski izvještaj:\n")
print(classification_report(y_test, y_test_p))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test,y_test_p))
disp.plot()
plt.title('Matrica zabune')
plt.show()



'''
Zadatak 0.0.3 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu otkrivanja dijabetesa, 
pri cemu je prvih 8 stupaca ulazna velicina, a u devetom stupcu se nalazi izlazna velicina: klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes). 
Ucitajte dane podatke. Podijelite ih na ulazne podatke  X i izlazne podatke y. 
Podijelite podatke na skup za ucenje i skup za testiranje modela u omjeru 80:20.
'''
'''
a) Izgradite neuronsku mrežu sa sljedecim karakteristikama: 
- model ocekuje ulazne podatke s 8 varijabli 
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
- izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
Ispišite informacije o mreži u terminal
'''
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

# OVAJ DIO NECE RADITI JER MI NE RADI TENSORFLOW TAKO JE NADALJE SAMO WISHFUL THINKING
model = keras.Sequential()
model.add(layers.Input(shape = (8,))) # ulazne znacajke
model.add(layers.Dense(12, activation="relu")) # prvi skriveni sloj
model.add(layers.Dense(8, activation="relu")) # drugi skriveni sloj
model.add(layers.Dense(1, activation="sigmoid")) # izlazni sloj
model.summary() # informacije o mrezi



'''
b) Podesite proces treniranja mreže sa sljedecim parametrima:
- loss argument: cross entropy
- optimizer: adam
- metrika: accuracy.
'''
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# c) Pokrenite ucenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i velicinom batch-a 10.
model.fit(x_train_vector, y_train_s, batch_size=10, epochs=150, validation_split=0.1)

# d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju ucitanog modela
model.save("model.keras")

# e) Izvršite evaluaciju mreže na testnom skupu podataka
from keras import models
model = models.load_model('model.keras')
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Testni gubitak (loss): {loss}")
print(f"Testna točnost (accuracy): {accuracy}")

# f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
# podataka za testiranje. Komentirajte dobivene rezultate.
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Matrica zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matrica zabune za testni skup")
plt.show()


'''
Matrica zabune prikazuje koliko je instanci model točno ili pogrešno klasificirao. Gornji lijevi kvadrant predstavlja 
točno negativne slučajeve (nema dijabetesa), dok donji desni kvadrant prikazuje točno pozitivne (ima dijabetesa). 
Preciznost i odziv pomažu razumjeti kako model balansira između lažno pozitivnih i lažno negativnih rezultata. 
Ako je odziv nizak, model propušta dijagnosticirati dijabetes, što može biti kritično.
'''