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
    data ['CO2 Emissions (g/km)'].plot(kind ='hist', bins = 20 )
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
    data.plot.scatter(
        x = 'Fuel Consumption City (L/100km)',
        y = 'CO2 Emissions (g/km)',
        cmap = 'hot', c = 'r', s = 50)
    plt.show()

def c_zadatak():
    '''
    c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske 
    potrošnje s obzirom na tip goriva. Primjecujete li grubu mjernu 
    pogrešku u podacima?
    '''