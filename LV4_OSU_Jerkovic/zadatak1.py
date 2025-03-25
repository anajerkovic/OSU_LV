import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
import sklearn.metrics as sk

data = pd.read_csv('data_C02_emission.csv')


'''
Odaberite željene numericke veličine specificiranjem liste s nazivima stupaca. Podijelite
podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%.
'''
input_var = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']
output = ['CO2 Emissions (g/km)']

X = data[input_var]
y = data[output]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =1 )



'''
Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova 
o jednoj numerickoj veličini. Pri tome podatke koji pripadaju skupu za učenje označite ˇ
plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
'''
plt.scatter(X_train['Engine Size (L)'], y_train, c ='Red')
plt.scatter(X_test['Engine Size (L)'], y_test, c='Blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()



'''
Izvršite standardizaciju ulaznih velicina skupa za učenje. Prikažite histogram vrijednosti 
jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja 
transformirajte ulazne velicine skupa podataka za testiranje.
'''
plt.hist(X_train['Fuel Consumption City (L/100km)'])
plt.show()

sc = MinMaxScaler ()
X_train_n = sc.fit_transform( X_train )
plt.hist(X_train_n[:,0])
plt.show()

X_test_n = sc.transform( X_test )



'''
Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.
'''
linearModel = lm.LinearRegression ()
linearModel.fit( X_train_n , y_train )

print(f"Parametri modela: {linearModel.coef_}")



'''
Izvršite procjenu izlazne velicine na temelju ulaznih veličina skupa za testiranje. Prikažite
pomocu dijagrama raspršenja odnos između stvarnih vrijednosti izlazne veličine i procjene 
dobivene modelom
'''
y_test_p = linearModel.predict ( X_test_n )
plt.scatter(y_test, y_test_p)
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predvidene vrijednosti")
plt.show()



'''
Izvršite vrednovanje modela na nacin da izračunate vrijednosti regresijskih metrika na 
skupu podataka za testiranje.
'''
MAE = sk.mean_absolute_error( y_test , y_test_p )
MSE = sk.mean_squared_error( y_test , y_test_p )
RMSE = sk.root_mean_squared_error( y_test , y_test_p )
MAPE = sk.mean_absolute_percentage_error( y_test , y_test_p )
R_squared = sk.r2_score( y_test , y_test_p )

print(f"MAE: {MAE}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"MAPE: {MAPE}")
print(f"R2: {R_squared}")


'''
Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj 
ulaznih velicina?
'''