import math
import numpy as np
import random
import json

listaData = []
listaData2 = []


# with open('australian.dat', 'r') as file:
#     for line in file:
#         listaData.append(list(map(lambda var: float(var), line.split())))
#         listaData2.append(list(map(lambda var: float(var), line.split())))


def metrykaEuk(x, y):
    wynik = 0
    for i in range(len(x)):
        wynik += math.pow(x[i] - y[i], 2)
    return math.sqrt(wynik)


def mierzymy(x, lista):
    my_dict = {0: [], 1: []}
    dl = max(len(x), len(lista[0])) - 1
    for i in range(len(lista)):
        my_dict[lista[i][dl]].append(metrykaEuk(x, lista[i]))
    return my_dict


x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def mierzymyTupki(x, lista):
    listaTupek = []
    dl = max(len(x), len(lista[0])) - 1
    for i in range(len(lista)):
        listaTupek.append((lista[i][dl], metrykaEuk(x, lista[i])))
    return listaTupek


def grupujemy(listaTupek):
    dictionary = {}
    for tupka in listaTupek:
        if tupka[0] not in dictionary.keys():
            dictionary[tupka[0]] = [tupka[1]]
        else:
            dictionary[tupka[0]].append(tupka[1])
    return dictionary


def najmniejszeK(slownikKlas, ile):
    for key in slownikKlas.keys():
        suma = 0
        lista = sorted(slownikKlas.get(key))
        for i in range(ile):
            # suma += slownikKlas.get(key)[i]
            suma += lista[i]
        slownikKlas[key] = suma
    return slownikKlas


def wybierzKlase(listaOdleglosci):
    odleglosc = listaOdleglosci[0]
    klasa = list(listaOdleglosci.keys())[0]
    for key in listaOdleglosci.keys():
        if listaOdleglosci.get(key) < odleglosc:
            odleglosc = listaOdleglosci.get(key)
            klasa = key

    listaOdleglosci[klasa] = -1
    if odleglosc in listaOdleglosci.values():
        return None

    return klasa


# print(mierzymyTupki(x, listaData))
# print(grupujemy(mierzymyTupki(x, listaData)))
# print(najmniejszeK(grupujemy(mierzymyTupki(x, listaData)), 5))
# print(wybierzKlase(najmniejszeK(grupujemy(mierzymyTupki(x, listaData)), 5)))

def metrykaEukSkalar(x, y):
    # x.pop()
    # y.pop()
    v1 = np.array(x)
    v2 = np.array(y)
    wektor = v2 - v1
    return math.sqrt(np.dot(wektor, wektor))


def pierwiastek(liczba):
    srednia = 1
    nowaLiczba = liczba
    for i in range(2000):
        srednia = float((nowaLiczba + srednia) / 2)
        nowaLiczba = float(liczba / srednia)
    return srednia


# print(pierwiastek(48671768))


listaDataBezKlasy = listaData
for i in range(len(listaDataBezKlasy)):
    listaDataBezKlasy[i].pop()


def kolorowanie(lista):
    ileKlas = 2  # na ile klas decyzyjnych dzielimy zbiór
    index_odleglosc = {}
    klasa_indexZListy = {}
    ileZmian = 0
    iteracja = 0
    odleglosc = 0
    posortowanySlownikOdleglosci = {}
    for i in range(len(lista)):
        lista[i].append(random.randint(0, ileKlas - 1))  # przypisz losowe klasy

    while True:
        iteracja += 1
        for klasa in range(ileKlas):  # sprawdzamy odleglosci do kazdej klasy
            for j in range(len(lista)):
                if lista[j][14] == klasa:
                    odleglosc = 0
                    for n in range(len(lista)):
                        if lista[n][14] == klasa:  # jeżeli punkt ma cały czas tą samą klasę
                            odleglosc += metrykaEuk(lista[j], lista[n])
                    index_odleglosc[j] = odleglosc  # przypisanie indexu z listy do odleglosci
            posortowanySlownikOdleglosci = dict(sorted(index_odleglosc.items(), key=lambda item: item[
                1]))  # posortowanie w celu znalezienia najmniejszej odleglosci
            klasa_indexZListy[klasa] = list(posortowanySlownikOdleglosci.keys())[
                0]  # wybranie tych najmniejszych z każdej klasy i przypisanie ich indexu z listy
            index_odleglosc = {}  # wyzerowanie slownika
            # print(klasa)
        print("główne punkty: " + json.dumps(klasa_indexZListy))

        for p in range(len(lista)):
            if p in klasa_indexZListy.values():  # nie iteruje się przez te elementy które są wybrane jako te centralne
                continue
            else:
                klasaDecyzyjna = lista[p][14]
                for klasa in range(ileKlas):
                    if metrykaEuk(lista[p], lista[klasa_indexZListy[klasa]]) < metrykaEuk(lista[p], lista[
                        klasa_indexZListy[
                            klasaDecyzyjna]]):  # sprawdź czy odległość od innej klasy jest bliższa niż obecna
                        klasaDecyzyjna = klasa
                        ileZmian += 1
                lista[p][14] = klasaDecyzyjna  # przypisuje nową klase dla pkt
        print("ilosc zmian: " + str(ileZmian))
        print("iteracjia: " + str(iteracja))
        if ileZmian == 0:
            break
        ileZmian = 0
    return lista


# nowaLista = kolorowanie(listaDataBezKlasy)
# klasa0 = []
# klasa1 = []
# for i in range(len(nowaLista)):
#     if nowaLista[i][14] == 0:
#         klasa0.append(nowaLista[i])
#     else:
#         klasa1.append(nowaLista[i])
#
# print(len(klasa0))
# print(len(klasa1))


def func1(x):
    return x


def monte_carlo(func, poziom, pion, ilosc_pkt):
    iloscPodWykresem = 0
    punkty_x = np.random.uniform(0, poziom, ilosc_pkt)
    punkty_y = np.random.uniform(0, pion, ilosc_pkt)
    punkty_y_wyliczone = [func(x) for x in punkty_x]

    for i in range(len(punkty_y)):
        if punkty_y[i] <= punkty_y_wyliczone[i]:
            iloscPodWykresem += 1

    pole = iloscPodWykresem / ilosc_pkt * poziom * pion

    return round(pole, 2)


def metodaProstokatowIlosc(iloscPodzialow, funkcja, poziom):
    suma = 0
    for i in range(1, iloscPodzialow + 1):
        x_0 = poziom / iloscPodzialow * (i - 1)
        x_1 = poziom / iloscPodzialow * i
        y_0 = funkcja(x_0)
        y_1 = funkcja(x_1)
        srednia = (y_0 + y_1) / 2
        suma += srednia
    return suma / iloscPodzialow


def metodaProstokatowEpsilon(epsilon, funkcja, poziom):
    suma1 = 0
    iloscPodzialow = 0
    wynik = math.inf
    while wynik >= epsilon:
        suma = 0
        iloscPodzialow += 10
        for i in range(1, iloscPodzialow + 1):
            x_0 = poziom / iloscPodzialow * (i - 1)
            x_1 = poziom / iloscPodzialow * i
            y_0 = funkcja(x_0)
            y_1 = funkcja(x_1)
            srednia = (y_0 + y_1) / 2
            suma += srednia
        if suma1 == 0:
            suma1 = suma
        else:
            wynik = (suma - suma1) / iloscPodzialow
            suma1 = suma

    return suma / iloscPodzialow


# print(f"Epsilon: {metodaProstokatowEpsilon(0.02, func1, 10)}")
# print(f"Prostokaty: {metodaProstokatowIlosc(1, func1, 10)}")
# print(f"Prostokaty: {metodaProstokatowIlosc(10, func1, 10)}")
# print(f"Prostokaty: {metodaProstokatowIlosc(1000, func1, 10)}")
# print(f"Prostokaty: {metodaProstokatowIlosc(10000, func1, 10)}")
# print(f"Monte Carlo: {monte_carlo(func1, 10, 10, 100)}")


def srednia_aryt(lista):
    suma = 0
    for elem in lista:
        suma += elem
    return suma / len(lista)


def wariancja(lista):
    srednia = srednia_aryt(lista)
    suma = 0
    for elem in lista:
        suma += (elem - srednia) ** 2
    return suma / len(lista)


def standard_odchyl(lista):
    return math.sqrt(wariancja(lista))


def regresja_liniowa(lista):
    X = np.array([[1, x[0]] for x in lista])
    Y = np.array([y[1] for y in lista])
    B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)


# lis = [(2, 1), (5, 2), (7, 3), (8, 3)]
# regresja_liniowa(lis)

def projekcja(u, v):
    up = np.dot(u, v)
    down = np.dot(u, u)
    if down == 0:
        down = 1
    return np.dot(up / down, u)


macierz_A = np.array([[0, 1], [3, 1], [1, 0]])


# macierz_A = np.array([
#     [0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 1],
#     [0, 1, 0, 0, 1],
#     [0, 0, 1, 0, 0],
#     [0, 0, 1, 0, 0]
# ])


def macierz_QR2(wektory):
    v = []
    e = []
    u = []
    for i in range(wektory.shape[1]):
        v.append(np.array([x[i] for x in wektory]))
        subtraction = v[i]
        if i < 1:
            u.append(v[0])
        else:
            for j in range(i):
                subtraction = np.subtract(subtraction, projekcja(u[j], v[i]))
            u.append(subtraction)
        len_u = metrykaEuk([0 for _ in range(wektory.shape[0])], u[i])
        if len_u == 0:
            len_u = 1
        e.append(u[i] / len_u)
    Q = np.array(e)
    R = np.dot(Q, wektory)
    # print(np.around(Q.T, decimals=2))
    # print(np.around(R, decimals=2))
    # print(np.around(np.dot(Q.T, R), decimals=1))
    return Q


def nowa_macierz(a):
    q = macierz_QR2(a)
    return np.dot(np.dot(q.T, a), q)


def wartosc_wlasna_macierzy(a):
    nowa_macierz_a = a
    while (np.diag(nowa_macierz_a) - np.dot(nowa_macierz_a, np.ones((nowa_macierz_a.shape[0], 1))).T).all() > 0.01:
        nowa_macierz_a = nowa_macierz(nowa_macierz_a)
    return np.diag(nowa_macierz_a)


def macierz_wlasnosci(matrix, wartosc):
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[0]):
            if x == y:
                matrix[x][y] = matrix[x][y] - wartosc

    return matrix


def gauss(matrix):
    column_len = len(matrix.T[0])

    for x in range(column_len):
        if matrix[x][x] == 0:
            matrix[x][x] = 1
            # raise ZeroDivisionError()

        for y in range(column_len):
            if x != y:
                factor = matrix[y][x] / matrix[x][x]

                for z in range(column_len + 1):
                    matrix[y][z] = matrix[y][z] - factor * matrix[x][z]
    return [matrix[i][column_len] / matrix[i][i] for i in range(column_len)]

# def gauss(matrix):
#     column_len = matrix.shape[0]
#
#     for y in range(column_len - 1):
#         for x in range(column_len - 1):
#             if y + x >= column_len - 1:
#                 continue
#             times = matrix[x + 1 + y][y] / matrix[y][y]
#             matrix[x + 1 + y] = matrix[x + 1 + y] - matrix[y] * times
#             print(matrix)
#
#     for y in range(column_len - 1):
#         for x in range(1, column_len):
#             if y + x > column_len - 1:
#                 continue
#             times = matrix[y][x + y] / matrix[x + y][x + y]
#             matrix[y] = matrix[y] - matrix[x + y] * times
#             print(matrix)
#
#     return [matrix[i][column_len] / matrix[i][i] for i in range(column_len)]


b = np.array([[1, 2, -1, 5],
              [3, 4, 1, 9],
              [2, -2, 3, -1]
              ])

b = np.array([[3, 2],
              [4, 1]
              ])

c = np.array([[3, 2],
              [4, 1]
              ])

print(wartosc_wlasna_macierzy(b))
for i in wartosc_wlasna_macierzy(b):
    b = c.copy()
    print(macierz_wlasnosci(b, np.round(i)))
    print(i)

# for i in range(b.shape[0]):
#     b[i].append(0)
#
# print(b)
print(gauss(np.append(b, np.zeros((b.shape[0], 1)), axis=1)))


# print(b)

b = np.array([[1, 3, 4],
              [2, 5, 8]
              ])

macierz_A = np.array([
    [4, -2, 4, -2, 8],
    [3, 1, 4, 2, 7],
    [2, 4, 2, 1, 10],
    [2, -2, 4, 2, 2]
])

# print(gauss(macierz_A))

matrix = np.array(
    [[1., 2., 3., 4., 5.], [2., 2., 3., 4., 5.], [3., 3., 3., 4., 5.], [4., 4., 4., 4., 5.], [5., 5., 5., 5., 5.]])

# qr = macierz_QR2(matrix)

# wynik = wartosc_wlasna_macierzy(matrix)
# print("Wynik", np.round(wynik, decimals=4), sep="\n")
