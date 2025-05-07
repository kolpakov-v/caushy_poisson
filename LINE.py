import numpy as np


def calculate_distances(x, y):
    # Проверяем, что массивы имеют одинаковую длину
    if len(x) != len(y):
        raise ValueError("Должны быть одинаковые размеры массивов.")

    # Вычисляем разности координат
    dx = np.diff(x)
    dy = np.diff(y)

    # Вычисляем расстояния
    distances = np.sqrt(dx ** 2 + dy ** 2)

    return distances

def complex_v_j():
    # G = [0.25, 0.5, 0.75, 1, 0.5, 0, -0.5, -1, -0.75, -0.5, -0.25]
    # n_o_p = 10
    n_o_p = 3
    x0_1 = np.linspace(-1, 0, n_o_p)
    y0_1 = -np.linspace(-1, 0, n_o_p)

    x0_2 = np.linspace(0, 1, n_o_p)[1:]
    y0_2 = np.linspace(0, 1, n_o_p)[1:]

    y0 = list(y0_1)
    y0.extend(list(y0_2))


    x0 = list(x0_1)
    x0.extend(list(x0_2))

    ## x0 and y0 it is coordinates of vortex

    return np.array(x0), np.array(y0)

def complex_line_for_shev_vesna_j():
    n1 = 15
    x0_1 = np.linspace(-3*np.pi, -(1/2)*np.pi, 15)
    y0_1 = np.zeros(15)

    n2 = 8
    t = np.linspace(np.pi, 2*np.pi, 8)
    x0_2 = np.cos(t)
    y0_2 = np.sin(t)

    n3 = 15
    x0_3 = np.linspace((1/2)*np.pi, 3*np.pi, 15)
    y0_3 = np.zeros(15)

    y0 = list(y0_1)
    y0.extend(list(y0_2))
    y0.extend(list(y0_3))

    x0 = list(x0_1)
    x0.extend(list(x0_2))
    x0.extend(list(x0_3))

    n_points = n1 + n2 + n3 - 2

    return np.array(x0), np.array(y0), n_points

def complex_line_for_comp_hydromech_j():
    n1 = 35
    # n1 = 25 # 58
    # n1 = 15 # 34
    n1 = 80
    x0_1 = np.linspace(-3*np.pi, -1, n1)[:-1]
    y0_1 = np.zeros(len(x0_1))

    n2 = int((2/5)*n1) + 150
    t = np.linspace(np.pi, 2*np.pi, n2)
    x0_2 = np.cos(t)
    y0_2 = np.sin(t)

    n3 = n1
    x0_3 = np.linspace(1, 3*np.pi, n3)[1:]
    y0_3 = np.zeros(len(x0_3))

    y0 = list(y0_1)
    y0.extend(list(y0_2))
    y0.extend(list(y0_3))

    x0 = list(x0_1)
    x0.extend(list(x0_2))
    x0.extend(list(x0_3))

    print(f'sum points aprox = {n1+n2+n3 - 2}')
    n_points = n1 + n2 + n3 - 2

    return np.array(x0), np.array(y0), n_points

def cosinus_line():
    n = 100
    x = np.linspace(-3*np.pi, 3*np.pi, n)
    y = (-1/2) * np.cos(x) -1

    return x, y

def cosinusoid_line():
    n1 = 30 #89
    n1 = 50 #150
    n1 = 15 #43
    # x1 = np.linspace(-3*np.pi, -np.pi, n1)[:-1]
    x1 = np.linspace(-1.5*np.pi, -np.pi, n1)[:-1]
    y1 = np.zeros(len(x1))
    # print(f'{calculate_distances(x1, y1)=}')

    l = 6.659
    n2 = int((n1*l)/(2*np.pi))
    print(n2)
    x2 = np.linspace(-np.pi, np.pi, n2)
    y2 = (-1/2) * np.cos(x2) -1/2
    # print(f'{calculate_distances(x2, y2)=}')

    n3 = n1
    # x3 = np.linspace(np.pi, 3*np.pi, n3)[1:]
    x3 = np.linspace(np.pi, 1.5 * np.pi, n3)[1:]
    y3 = np.zeros(len(x3))
    # print(f'{calculate_distances(x3, y3)=}')

    y0 = list(y1)
    y0.extend(list(y2))
    y0.extend(list(y3))

    x0 = list(x1)
    x0.extend(list(x2))
    x0.extend(list(x3))

    print(f'sum points aprox = {n1+n2+n3 - 2}')
    n_points = n1+n2+n3 - 2
    return np.array(x0), np.array(y0), n_points

def cosinusoid_line_longer():
    # n1 = 130 #313
    # n1 = 50 #119
    n1 = 35 #82

    n1 = 130

    x1 = np.linspace(-6*np.pi, -np.pi, n1)[:-1]
    y1 = np.zeros(len(x1))
    # print(f'{calculate_distances(x1, y1)=}')

    l = 6.659
    n2 = int((n1*l)/(5*np.pi))
    print(n2)
    x2 = np.linspace(-np.pi, np.pi, n2)
    y2 = (-1/2) * np.cos(x2) -1/2
    # print(f'{calculate_distances(x2, y2)=}')

    n3 = n1
    x3 = np.linspace(np.pi, 6*np.pi, n3)[1:]
    y3 = np.zeros(len(x3))
    # print(f'{calculate_distances(x3, y3)=}')

    y0 = list(y1)
    y0.extend(list(y2))
    y0.extend(list(y3))

    x0 = list(x1)
    x0.extend(list(x2))
    x0.extend(list(x3))

    print(f'sum points aprox = {n1+n2+n3 - 2}')
    n_points = n1+n2+n3 - 2
    return np.array(x0), np.array(y0), n_points