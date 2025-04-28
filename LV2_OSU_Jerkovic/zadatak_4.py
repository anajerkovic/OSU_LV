import numpy as np
import matplotlib.pyplot as plt

'''
Napišite program koji ce kreirati sliku koja sadrži cetiri kvadrata crne odnosno ˇ
bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili
u odgovarajuci oblik koristite numpy funkcije hstack i vstack.
'''

ones = np.ones((50,50))
zeros = np.zeros((50,50))

top = np.hstack((zeros, ones))
bottom = np.hstack((ones, zeros))
matrix = np.vstack((top, bottom))

plt.figure()
plt.imshow(matrix, cmap="gray")
plt.show()