import numpy as np
import matplotlib.pyplot as plt

'''
Zadatak 2.4.2 Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja  data pri cemu je u ˇ
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci´
stupac polja je masa u kg.
'''
file = open("data.csv")
data = np.loadtxt(file, skiprows=1, delimiter=',')
file.close()

#a) Na koliko je osoba izvršeno mjerenje
print(f"a) {data.shape}")

#b) Odnos visine i mase osobe
plt.scatter(data[:,1], data[:,2],)
plt.title("b) Odnos visine i mase osobe")
plt.show()

#c) Odnos visine i mase za 50. osobu
plt.scatter(data[::50,1], data[::50,2],)
plt.title("c) Odnos visine i mase za svaku 50. osobu")
plt.show()

#d)
print(f"Minimalna vrijednost visine: {data[:,1].min()}")
print(f"Maksimalna vrijednsost visine: {data[:,1].max()}")
print(f"Srednja vrijednost visine: {data[:,1].mean()}")

#e)
indM = (data[:,0] == 1) # svi retci koji su muški (indeksi)
indW = (data[:,0] == 0) # svi retci koji su ženski

print(data[indM,1].min())
print(data[indM,1].max())
print(data[indM,1].mean())


print(data[indW,1].min())
print(data[indW,1].max())
print(data[indW,1].mean())
