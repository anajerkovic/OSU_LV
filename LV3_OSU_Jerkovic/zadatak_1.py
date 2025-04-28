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
    data.dropna() # brisanje null vrijednosti
    data['Vehicle Class'] = data['Vehicle Class'].astype('category') # kategoricki tipovi -- category tip
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
      prosječna emisija C02 plinova automobila proizvodača Audi koji imaju 4 cilindara?
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
    even_cylinders = even_cylinders.groupby('Cylinders') # moram grupirati kako bi mogla odrediti mean (srednju vrijednost)
    print(even_cylinders['CO2 Emissions (g/km)'].mean())


def f_zadatak():
    '''
    f) Kolika je prosjecna gradska potrošnja u sluČaju vozila koja 
    koriste dizel, a kolika za vozila 
    koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
    '''
    diesels = data[(data['Fuel Type'] == 'D')]
    petrols = data[(data['Fuel Type'] == 'Z')]

    print(f"Dizeli:\nProsjecno: {diesels['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {diesels['Fuel Consumption City (L/100km)'].median()}")
    print(f"Benzinci:\nProsjecno: {petrols['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {petrols['Fuel Consumption City (L/100km)'].median()}")

def g_zadatak():
    '''
    Koje vozilo s 4 cilindra koje koristi dizelski 
    motor ima najvecu gradsku potrošnju goriva?
    '''
    car = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == "D")]
    print(car.nlargest(1, 'Fuel Consumption City (L/100km)'))

def h_zadatak():
    '''
    Koliko ima vozila ima rucni tip mjenjača (bez obzira na broj brzina)?
    '''
    manual_transsmission = data[(data['Transmission'].str.startswith('M'))]
    print(len(manual_transsmission['Transmission']))

def i_zadatak():
    '''
    Izracunajte korelaciju između numeričkih veličina. 
    Komentirajte dobiveni rezultat
    '''
    print (data.corr( numeric_only = True ))
'''
Komentiranje zadnjeg zadatka:
Velicine imaju dosta veliki korelaciju. Npr. broj obujam motora i broj cilindara su oko 0.9, dok je potrosnja oko 0.8 sto ukazuje na veliku korelaciju.
Takodjer razlog zasto potrosnja u mpg ima veliku negativnu korelaciju je to sto je ta velicina obrnuta, odnosno, sto automobil vise trosi, broj je manji
Npr: automobil koji trosi 25 MPG trosi vise nego automobil koji trosi 45 MPG. Dakle, ta velicina je obrnuta L/100km te takodjer, zbog toga dobivamo negativnu
korelaciju. Sto je negativna korelacija blize -1 to je ona vise obrnuto proporcijalna, dok sto je blize 1, to je vise proporcijonalna. Vrijednosti oko 0
nemaju nikakvu korelaciju s velicinom.
'''