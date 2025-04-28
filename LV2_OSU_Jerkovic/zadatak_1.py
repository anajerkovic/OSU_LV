import numpy as np
import matplotlib.pyplot as plt

'''
Zadatak 2.4.1 Pomocu funkcija numpy.array i matplotlib.pyplot pokušajte nacrtati sliku
2.3 u okviru skripte zadatak_1.py. Igrajte se sa slikom, promijenite boju linija, debljinu linije i
sl.
'''

a = np.linspace(1,3, num= 2) # od 1 do 3 na x osi
b = np.linspace(1,1, num = 2) # od 1 do 1 na y osi
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
plt.title ('Zadatak 1 - svaka crta posebno')
plt.show()

x = np.array([1,3,3,2,1])
y = np.array([1,1,2,2,1])
plt.plot(x,y,'magenta',linewidth=2, marker=".")
plt.axis ([0 ,4 ,0 , 4])
plt.xlabel ('x os')
plt.ylabel('y os')
plt.title ('Zadatak 1 - sve odjednom')
plt.show()