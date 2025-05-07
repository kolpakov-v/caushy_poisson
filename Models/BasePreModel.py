import numpy as np
import FUNCTIONS as f

class BaseModel:

    ''' --------------- quiver plot --------------- '''
    def V(self, X, Y, G0, x0, y0, dt):
        delta = 0.01
        def velocity(G0, x0, y0, x, y):
            R = max(delta, np.sqrt((2 * np.pi * ((x - x0) ** 2 + (y - y0) ** 2))))
            k = G0 / R**2
            u = k * (y0 - y)
            v = k * (x - x0)

            return (u, v)

        x_vec, y_vec = [[] for _ in range(len(X))], [[] for _ in range(len(X))]

        for i in range(len(X)):
            for j in range(len(X[0])):
                u, v = 0, 0
                for k in range(0, len(G0)):
                    x0_p = x0[k]
                    y0_p = y0[k]

                    x = X[i][j]
                    y = Y[i][j]

                    if isinstance(G0, list):
                        u += velocity(G0[k], x0_p, y0_p, x, y)[0]
                        v += velocity(G0[k], x0_p, y0_p, x, y)[1]
                    else:
                        u += velocity(G0, x0_p, y0_p, x, y)[0]
                        v += velocity(G0, x0_p, y0_p, x, y)[1]

                x_vec[i].append(u)
                y_vec[i].append(v)

        return np.array(x_vec), np.array(y_vec)

    ''' ------ caushy poisson ------ '''

    def velocity(self, G0, x0, y0, x, y):
        # delta = 0.01
        delta = 0.01
        # print(f'type(G0) in velocity = {type(G0)}')
        if isinstance(G0, np.ndarray) or isinstance(G0, list):
            u, v = 0, 0
            for i in range(0, len(G0)):
                R = max(delta, np.sqrt(((x - x0[i]) ** 2 + (y - y0[i]) ** 2)))

                k = G0[i] / (2 * np.pi * R ** 2)
                u = u + k * (y0[i] - y)
                v = v + k * (x - x0[i])
        else:
            R = max(delta, np.sqrt(((x - x0) ** 2 + (y - y0) ** 2)))
            k = G0 / (2 * np.pi * R**2)
            u = k * (y0 - y)
            v = k * (x - x0)

        return [u, v]

    def init_G(self, x0, y0):
        xk = (x0[1:] + x0[:-1]) / 2
        yk = (y0[1:] + y0[:-1]) / 2
        r = np.sqrt((x0[1:] - x0[:-1])**2 + (y0[1:] - y0[:-1])**2)

        nxk =  (y0[1:] - y0[:-1]) / r
        nyk =  - (x0[1:] - x0[:-1]) / r

        A = np.array([])
        M = len(x0)
        for i in range(0, M-1):
            for j in range(0, M):
                R = max(0.01, np.sqrt(((xk[i] - x0[j]) ** 2 + (yk[i] - y0[j]) ** 2)))
                k = 1 / (2 * np.pi * R ** 2)
                u = k * (y0[j] - yk[i])
                v = k * (xk[i] - x0[j])
                val = nxk[i]*u + nyk[i]*v

                A = np.append(A, val)
        A = np.append(A, [1 for _ in range(0, M)])

        A = A.reshape((M, M))
        b = np.array([0 for _ in range(0, M)])

        G0 = np.linalg.solve(A, b)
        return G0

    def new_G(self, x0_new, y0_new, x0, y0, G, dt, G0 = 0):

        xs_new = (x0_new[1:] + x0_new[:-1]) / 2
        ys_new = (y0_new[1:] + y0_new[:-1]) / 2

        xs = (x0[1:] + x0[:-1]) / 2
        ys = (y0[1:] + y0[:-1]) / 2

        ''' right side of equation system '''
        A = np.array([])

        for s in range(0, len(xs)):
            for i in range(0, len(G)):
                val = f.ARCTG(xs_new[s], ys_new[s], x0_new[i], y0_new[i])/(2*np.pi)
                A = np.append(A, val)

        A = np.append(A, [1 for _ in range(0, len(G))])
        M = len(G)
        A = A.reshape((M, M))

        ''' left side of equation system '''

        b = []
        for s in range(0, len(xs)):
            val = 0
            V_2 = self.velocity(G, x0, y0, xs[s], ys[s])[0] ** 2 + self.velocity(G, x0, y0, xs[s], ys[s])[1] ** 2
            for i in range(0, len(G)-1):
                Fr = 1
                # Fr = 3
                val += G[i]*f.ARCTG(xs[s], ys[s], x0[i], y0[i])/(2*np.pi)
            val += (V_2/2 - ys[s]/Fr)*dt

            b.append(val)
        b.append(sum(G))
        G_new = np.linalg.solve(A, np.array(b))
        return G_new

        if np.linalg.cond(A) < 1000:
            G_new = np.linalg.solve(A, np.array(b)) #numpy
        else:
            from sklearn.linear_model import Ridge
            from sklearn.linear_model import Lasso

            ridge = Ridge(alpha=150) #0.1 brake #150 #alpha=100 #alpha=10 #alpha=5
            ridge.fit(A, b)

            G_new = ridge.coef_

        # bad
        # lasso = Lasso(alpha=0.001)
        # lasso.fit(A, b)
        #
        # G_new = lasso.coef_

        return G_new
