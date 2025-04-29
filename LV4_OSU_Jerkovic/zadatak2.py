import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn.linear_model as lm
import sklearn.metrics as sk

data = pd.read_csv('data_C02_emission.csv')
data.info()
'''
Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku 
varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategoričkih 
velicina. 
Radi jednostavnosti nemojte skalirati ulazne veličine - no minmax. Komentirajte dobivene rezultate. 
Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
vozila radi?
'''

#ovaj dio je isti kao i u 1. zadatku
input_var = ['Engine Size (L)', 
             'Cylinders', 
             'Fuel Type', 
             'Fuel Consumption City (L/100km)',
             'Fuel Consumption Hwy (L/100km)',
             'Fuel Consumption Comb (L/100km)',
             'Fuel Consumption Comb (mpg)',
             ]
output_var = ['CO2 Emissions (g/km)']
X = data[input_var]
y = data[output_var]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)


#Kodiranje nominalne kategoricke velicine Fuel Type - one hot encoding
ohe = OneHotEncoder()
X_encoded_train = ohe.fit_transform(X_train[['Fuel Type']]).toarray()
X_encoded_test = ohe.fit_transform(X_test[['Fuel Type']]).toarray()

# izgrađivanje linearnog regresijskog modela
linearModel = lm.LinearRegression()
linearModel.fit(X_encoded_train, y_train)
y_test_p = linearModel.predict(X_encoded_test)

#izračun maksimalne pogreške ME
ME = sk.max_error(y_test, y_test_p)
print(f"Max Error: {ME}") # 331.45


error = np.abs(y_test_p, y_test)
max_error_id = np.argmax(error)

max_error_model = data.iloc[max_error_id, 1]
print(max_error_model)