import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

img = plt.imread("road.jpg")
imga = img [:,:,0].copy()
plt.figure()
plt.imshow( imga , cmap ="gray", alpha=0.5)
plt.show()

imgb = img [:,160:320,0].copy()
plt.figure()
plt.imshow( imgb , cmap ="gray")
plt.show()

imgc = np.rot90(img,3)
plt.figure()
plt.imshow( imgc , cmap ="gray")
plt.show()

