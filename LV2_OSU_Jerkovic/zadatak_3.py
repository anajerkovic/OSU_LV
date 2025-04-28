import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

'''
Skripta zadatak_3.py ucitava sliku ’ ˇ road.jpg’. Manipulacijom odgovarajuce´
numpy matrice pokušajte:
'''

# a) posvijetliti sliku
img = plt.imread("road.jpg")
imga = img [:,:,0].copy()
plt.figure()
plt.imshow( imga , cmap ="gray", alpha=0.5)
plt.show()


# b) prikazati samo drugu četvrtinu slike po širini
imgb = img [:,160:320,0].copy() # slika se sastoji od 640 stupaca
plt.figure()
plt.imshow( imgb , cmap ="gray")
plt.show()

# c) zarotirati sliku 90 stupnjeva
imgc = np.rot90(img,3) # numpy biblioteka ima funkciju rot90
plt.figure()
plt.imshow( imgc , cmap ="gray")
plt.show()


# d) zrcaliti sliku
imgd = np.fliplr(img) #opet numpy funkcija
plt.figure()
plt.imshow( imgd , cmap ="gray")
plt.show()
