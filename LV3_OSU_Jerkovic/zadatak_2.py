import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data_C02_emission.csv')

def a_zadatak():
    '''
    Pomocu histograma prikažite emisiju C02 plinova. 
    Komentirajte dobiveni prikaz.
    '''
    plt.figure ()
    data['CO2 Emissions (g/km)'].plot(kind ='hist', bins = 20)
    plt.title("Emisija CO2 plinova")
    plt.show()


def b_zadatak():
    '''
    b) Pomocu dijagrama raspršenja prikažite odnos između gradske 
    potrošnje goriva i emisije 
    C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli 
    odnose izmedu
    velicina, obojite točkice na dijagramu raspršenja s 
    obzirom na tip goriva.
    '''
    data['Fuel Type'] = data['Fuel Type'].astype('category')
    colors = {'Z': 'green', 'X': 'red', 'E': 'blue', 'D': 'black'}

    data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c=data["Fuel Type"].map(colors), s=50)
    plt.show()


def c_zadatak():
    '''
    c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske 
    potrošnje s obzirom na tip goriva. Primjecujete li grubu mjernu 
    pogrešku u podacima?
    '''
    data.boxplot(column='CO2 Emissions (g/km)', by='Fuel Type')
    plt.show()

def d_zadatak():
    '''
    Pomocu stupcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
    groupby.
    '''
    fuel_grouped_num = data.groupby('Fuel Type').size()
    fuel_grouped_num.plot(kind ='bar', xlabel='Fuel Type', ylabel='Number of vehicles', title='Amount of vehicles by fuel type')
    plt.show()

def e_zadatak():
    '''
    Pomocu stupčastog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
    broj cilindara.
    '''
    cylinder_grouped = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
    cylinder_grouped.plot(kind='bar', x=cylinder_grouped.index, y=cylinder_grouped.values, xlabel='Cylinders', ylabel='CO2 emissions (g/km)', title='CO2 emissions by number of cylinders')
    plt.show()

e_zadatak()