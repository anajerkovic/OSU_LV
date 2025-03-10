import numpy as np
import matplotlib.pyplot as plt


file = open("data.csv")
data = np.loadtxt(file,skiprows=1,delimiter=",")

file.close()
print(data.shape)

#plt.scatter(data[:,1], data[:,2])
#plt.show()

plt.scatter(data[::50,1], data[::50,2])
plt.show()

print("d)")
print(data[:,1].min())
print(data[:,1].max())
print(data[:,1].mean())


print("e)")
ind = (data[:,0] == 1)
print(data[ind,1].min())
print(data[ind,1].max())
print(data[ind,1].mean())

indF = (data[:,0] == 0)
print(data[indF,1].min())
print(data[indF,1].max())
print(data[indF,1].mean())