import numpy as np
np.set_printoptions(precision=4)


def ARCTG(x, y, x0, y0):
    '''

    :param y:
    :param y0:
    :param x:
    :param x0:
    :return: int num
    '''

    x2 = y
    y2 = -x

    x02 = y0
    y02 = -x0

    if (x2 == x02 and y2 == y02):
        # print(1)
        return 0
    elif x2 - x02 > 0 and y2 - y02 >= 0:
        # print(2)
        k = (y2 - y02) / (x2 - x02)
        return np.arctan(k)
    elif x2 - x02 <= 0 and y2 - y02 > 0:
        # print(3)
        # k = (y - y0) / (x - x0)
        t = (x2 - x02) / (y2 - y02)
        return (np.pi / 2 - np.arctan(t))
    elif x2 - x02 < 0 and y2 - y02 <= 0:
        # print(4)
        k = (y2 - y02) / (x2 - x02)
        return (np.pi + np.arctan(k))
    elif x2 - x02 >= 0 and y2 - y02 < 0:
        # print(5)
        # k = (y - y0) / (x - x0)
        t = (x2 - x02) / (y2 - y02)
        return (3 / 2) * np.pi - np.arctan(t)

def phi_dipol(X, Y, G, x0, y0):
    delta = 0.1
    arr_val = [[] for _ in range(len(X))]
    # for i in tqdm(range(len(X))):
    for i in range(len(X)):
        for k in range(len(X[i])):
            x = X[i][k]
            y = Y[i][k]

            val = 0

            for j in range(len(G) - 1):
                # print(f'j = {j}')
                sum_G = sum(G[:j + 1])

                xj = (x0[j + 1] + x0[j]) / 2
                yj = (y0[j + 1] + y0[j]) / 2

                # R = max(0.2, np.sqrt((x-xj)**2 + (y-yj)**2))
                R = max(delta, (x - xj) ** 2 + (y - yj) ** 2)
                t = ((y0[j + 1] - y0[j]) * (x - xj) - (x0[j + 1] - x0[j]) * (y - yj)) / R

                val = val + (sum_G / (2 * np.pi)) * t
            arr_val[i].append(val)
    return_val = arr_val

    return return_val

def pci_dipol(X, Y, G, x0, y0):
    delta = 0.2
    arr_val = [[] for _ in range(len(X))]
    # for i in tqdm(range(len(X))):
    for i in range(len(X)):
        for k in range(len(X[i])):
            x = X[i][k]
            y = Y[i][k]

            val = 0
            for j in range(len(G) - 1):
                sum_G = sum(G[:j + 1])

                xj = (x0[j + 1] + x0[j]) / 2
                yj = (y0[j + 1] + y0[j]) / 2

                R = max(delta, (x - xj) ** 2 + (y - yj) ** 2)
                t = ((x0[j + 1] - x0[j]) * (x - xj) + (y0[j + 1] - y0[j]) * (y - yj)) / R

                val = val + (sum_G / (2 * np.pi)) * t

            val = -1 * val
            arr_val[i].append(val)
    return_val = arr_val

    return return_val

def phi_ARCTG(X, Y, G, x0, y0):
    '''

    :param x:
    :param y:
    :param G: list
    :param y0: list, y coordinate of whistl...
    :param x0: list, x coordinate of whistl...
    :return: value
    '''

    arr_val = [[] for _ in range(len(X))]
    # for i in tqdm(range(len(X))):
    for i in range(len(X)):
        for k in range(len(X[i])):
            val = 0
            for j in range(len(G)):
                # (x, y, x0, y0)
                # val += G[j]*ARCTG(Y[i][k], y0[j], X[i][k], x0[j])/(2*np.pi)
                val += G[j] * ARCTG(X[i][k], Y[i][k], x0[j], y0[j]) / (2 * np.pi)
            arr_val[i].append(val)
    return_val = arr_val

    return return_val

def pci_ARCTG(X, Y, G, x0, y0):
    arr_val = [[] for _ in range(len(X))]
    for i in range(len(X)):
        for k in range(len(X[i])):
            val = 0
            for j in range(len(G)):
                r = np.sqrt((X[i][k] - x0[j]) ** 2 + (Y[i][k] - y0[j]) ** 2)
                if r == 0:
                    r = 1e-10
                val += G[j] * np.log(r) / (2 * np.pi)
            arr_val[i].append(-val)
    return_val = arr_val

    return return_val