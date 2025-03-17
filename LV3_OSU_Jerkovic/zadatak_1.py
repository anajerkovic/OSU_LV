import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

def a_zadatak():
    '''
    a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? 
    Postoje li izostale ili  duplicirane vrijednosti? 
    Obrišite ih ako postoje. 
    Kategoricke veličine konvertirajte u tip 
    category
    '''
    print(len(data)) # 2212 mjerenja
    print(data.info()) # int, float, object
    data.drop_duplicates() # nema dupliciranih vrijednosti
    data.dropna()
    data['Vehicle Class'] = data['Vehicle Class'].astype('category')
    print(data.info())


def b_zadatak():
    '''
    b) Koja tri automobila ima najvecu odnosno najmanju 
    gradsku potrošnju? Ispišite u terminal: 
    ime proizvodača, model vozila i kolika je gradska potrošnja
    '''

    least_consumption = data.nsmallest(3, 'Fuel Consumption City (L/100km)')
    print(least_consumption[['Make','Model','Fuel Consumption City (L/100km)']])
    biggest_consumption = data.nlargest(3, 'Fuel Consumption City (L/100km)')
    print(biggest_consumption[['Make','Model','Fuel Consumption City (L/100km)']])

def c_zadatak():
    '''
    c) Koliko vozila ima velicinu motora između 2.5 i 3.5 L? 
    Kolika je prosječna C02 emisija
    plinova za ova vozila?
    '''
    co2_data = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
    print(co2_data)
    print(f"Prosjecna CO2 emisija: {co2_data['CO2 Emissions (g/km)'].mean()}")

def d_zadatak():
    '''
    d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? Kolika je
      prosječna emisija C02 
    plinova automobila proizvodača Audi koji imaju 4 cilindara?
    '''
    audis = data[(data['Make'] == "Audi")]
    print(f"Audi mjerenja: {len(audis)}")

    audis = audis[(audis['Cylinders'] == 4)]
    print(f"Prosjecna CO2 emisija: {audis['CO2 Emissions (g/km)'].mean()}")

def e_zadatak():
    '''
    e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna 
    emisija C02 plinova s obzirom na ˇ
    broj cilindara? 
    '''
    even_cylinders = data[(data['Cylinders'] % 2 == 0)]
    print(f"Broj auto sa parnim brojem cilindara: {len(even_cylinders)}")
    even_cylinders = even_cylinders.groupby('Cylinders')
    print(even_cylinders['CO2 Emissions (g/km)'].mean())


def f_zadatak():
    '''
    f) Kolika je prosjecna gradska potrošnja u sluČaju vozila koja 
    koriste dizel, a kolika za vozila 
    koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
    '''
    fuel_data = data.groupby('Fuel Type')
    print(fuel_data['Fuel Consumption City (L/100km)'].mean())
    print(fuel_data['Fuel Consumption City (L/100km)'].median())

def g_zadatak():
    '''
    Koje vozilo s 4 cilindra koje koristi dizelski 
    motor ima najvecu gradsku potrošnju goriva?
    '''
    car = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == "D")]
    print(car.nlargest(1, 'Fuel Consumption City (L/100km)'))

def h_zadatak():
    '''
    h) Koliko ima vozila ima rucni tip mjenjača (bez obzira na broj brzina)?
    '''
    manual_transsmission = data[(data['Transmission'].str.startswith('M'))]
    print(len(manual_transsmission['Transmission']))

def i_zadatak():
    '''
    Izracunajte korelaciju između numeričkih veličina. 
    Komentirajte dobiveni rezultat
    '''
    print (data.corr( numeric_only = True ))

i_zadatak()