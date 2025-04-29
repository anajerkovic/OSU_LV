'''
Skripta zadatak_1.py ucitava podatkovni skup iz data_C02_emission.csv.
Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih veličina. 
Detalje oko ovog podatkovnog skupa mogu se pronaći u 3. 
laboratorijskoj vježbi.
'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
import sklearn.metrics as sk

data = pd.read_csv('data_C02_emission.csv')
data.info()

'''
Odaberite željene numericke veličine specificiranjem liste s nazivima stupaca. Podijelite
podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%.
'''
input_var = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 
             'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']
output = ['CO2 Emissions (g/km)']
X = data[input_var]
y = data[output]

# podjela skupa na skup za ucenje i skup za testiranje 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =1 )



'''
Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova 
o jednoj numerickoj veličini(Engine Size). Pri tome podatke koji pripadaju skupu za učenje označite 
plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
'''
plt.scatter(X_train['Engine Size (L)'], y_train, c ='Blue', label='Training data')
plt.scatter(X_test['Engine Size (L)'], y_test, c='Red', label='Testing data')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.show()



'''
Izvršite standardizaciju ulaznih velicina skupa za učenje. 
Prikažite histogram vrijednosti jedne ulazne velicine(Fuel Consumption City (L/100km)) prije i nakon skaliranja. 
Na temelju dobivenih parametara skaliranja transformirajte ulazne velicine skupa podataka za testiranje.
'''
plt.subplot(1,2,1)
plt.hist(X_train['Fuel Consumption City (L/100km)'])
plt.title("Prije Standardizacije")
plt.tight_layout()

scaler = MinMaxScaler()
X_train_n = scaler.fit_transform( X_train )

plt.subplot(1,2,2)
plt.hist(X_train_n[:,0])
plt.title("Poslije Standardizacije")
plt.tight_layout()
plt.show()

X_test_n = scaler.transform( X_test )



'''
Izgradite linearni regresijski modeli. 
Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.
'''
linearModel = lm.LinearRegression ()
linearModel.fit( X_train_n , y_train ) #procjena parametara na temelju skupa za učenje/treniranje

print(f"Parametri modela: {linearModel.coef_}")



'''
Izvršite procjenu izlazne velicine na temelju ulaznih veličina skupa za testiranje. 
Prikažite pomocu dijagrama raspršenja odnos između stvarnih vrijednosti izlazne veličine i procjene 
dobivene modelom
'''
y_test_p = linearModel.predict ( X_test_n ) # predikcija izlazne veličine na skupu podataka za testiranje
plt.scatter(y_test, y_test_p)
plt.title("Odnos između stvarnih vrijednosti \n izlazne veličine i procjene dobivene modelom")
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Procjenjene vrijednosti")
plt.show()



'''
Izvršite vrednovanje modela na nacin da izračunate vrijednosti regresijskih metrika na 
skupu podataka za testiranje.Sve funkcije su dio scikit-learn biblioteke
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