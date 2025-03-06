#zadatak 1.4.1
def total_euro():
    hours = int(input("Radni sati: "))
    pay = float(input("eura/h: "))
    total = hours * pay
    print(f"Ukupno: {total} eura")

#total_euro()

#zadatak 1.4.2
def grades():
    try:
        grade = float(input("Unesi ocjenu od 0.0 do 1.0: "))
    except:
        print("Krivi unos")

    if grade < 0.0 or grade > 1.0:
        print("Ocjena nije u granicama.")
    else:
        if grade >= 0.9:
            print("A")
        elif grade >= 0.8:
            print("B")
        elif grade >= 0.7:
            print("C")
        elif grade >= 0.6:
            print("D")
        else:
            print("F")

#grades()

#zadatak 1.4.3

def numbers():
    num_list = []
    num_input = input("Input a number: ")
    while (num_input != "Done"):
        if (num_input.isnumeric()):
            num_list.append(int(num_input))
        else:
            print("Not a number")
        num_input = input("Input a number: ")

    print(len(num_list))
    print(min(num_list))
    print(max(num_list))
    average = sum(num_list)/len(num_list)
    print(average)

#numbers()


#Zadatak 1.4.4

def words_in_songs():
    all_words = {}
    fhand = open('song.txt')
    for line in fhand:
        for word in line.split():
            if word not in all_words:
                all_words[word] = 1
            else:
                all_words[word] += 1
    print(all_words)
    fhand.close

    for key in all_words.keys():
        if all_words[key] == 1:
            print(key)

#words_in_songs()


#1.4.5
def ham_or_spam():
    hams = 0
    spams = 0
    hams_sum = 0
    spams_sum = 0
    exclamations = 0
    file = open('SMSSpamCollection.txt')
    for line in file:
        curr_element = line.split()
        if (curr_element[0]  == "ham"):
            hams += 1
            hams_sum += len(curr_element)
        elif (curr_element[0]  == "spam"):
            spams += 1
            spams_sum += len(curr_element)
            if(curr_element[-1][-1] == "!"):
                exclamations += 1
    file.close()

    print(hams_sum/hams)
    print(spams_sum/spams)
    print(exclamations)




ham_or_spam()
