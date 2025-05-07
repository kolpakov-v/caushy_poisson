from Models.BasePreModel import BaseModel

import numpy as np


class Model_wNormal(BaseModel):

    def normal(self, x0, y0):
        xk = (x0[1:] + x0[:-1]) / 2
        yk = (y0[1:] + y0[:-1]) / 2
        r = np.sqrt((xk[1:] - xk[:-1]) ** 2 + (yk[1:] - yk[:-1]) ** 2)

        nxk = (yk[1:] - yk[:-1]) / r
        nyk = - (xk[1:] - xk[:-1]) / r

        return nxk, nyk

    def distances(self, x0, y0):
        dist_x = x0[1:] - x0[:-1]
        dist_y = y0[1:] - y0[:-1]

        return dist_x, dist_y

    # тут ми рахуємо швидкість не в точкі вихору, а відходимо на нормаль
    def new_surface_2(self, x0, y0, G, x, y, dt):
        x_new = []
        y_new = []

        nxk, nyk = self.normal(x0, y0)

        nxk = np.insert(nxk, 0, nxk[0])
        nyk = np.insert(nyk, 0, nyk[0])

        nxk = np.insert(nxk, len(nxk), nxk[-1])
        nyk = np.insert(nyk, len(nyk), nyk[-1])

        # normal delta
        # delta = 0.01 # in dipol delta = 0.1
        # delta = 0.5
        # # delta = 0.1
        # delta = 0.01
        # # delta = 0
        from main import normal_delta
        delta = normal_delta

        for i in range(0, len(x)):
            u, v = 0, 0
            for j in range(0, len(G)):
                vel = self.velocity(G[j], x0[j], y0[j], x[i] + nxk[i] * delta, y[i] + nyk[i] * delta)
                u = u + vel[0]
                v = v + vel[1]

            xn = x[i] + u * dt
            yn = y[i] + v * dt

            x_new.append(xn)
            y_new.append(yn)

        return np.array(x_new), np.array(y_new), nxk, nyk
