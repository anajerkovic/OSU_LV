import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(1,3, num= 2)
b = np.linspace(1,1, num = 2)
plt.plot(a,b,'b', linewidth=2, marker=".")

c = np.linspace(1,2,num=2)
d= np.linspace(1,2,num=2)
plt.plot(c,d,'r', linewidth=2, marker=".")

e=np.linspace(2,3,num=2)
f = np.linspace(2,2,num=2)
plt.plot(e,f,'g', linewidth=2, marker=".")

g = np.linspace(3,3,num = 2)
h = np.linspace(1,2,num = 2)
plt.plot(g,h,'y', linewidth=2, marker=".")

plt.axis ([0 ,4 ,0 , 4])
plt.xlabel ('x os')
plt.ylabel('y os')
plt.title ('Zadatak 1')
plt.show()