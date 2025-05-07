import FUNCTIONS as f

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from tqdm import tqdm
from fractions import Fraction
np.set_printoptions(precision=4)

class Draw:
    def __init__(self, model, X, Y, N, dt, normal_delta, l_ax_x, l_ax_y, l_grid, n_points_grid,
                 PICS_DIR_NAME='with_normal',
                 is_draw_phi_dipol=False,
                 is_draw_pci_dipol=False,
                 is_draw_phi_ARCTG=False,
                 is_draw_pci_ARCTG=False,
                 is_draw_quiver=False,
                 is_draw_quiver_in_points=False,
                 show_pics=False,
                 save_pics_and_params=False,
                 ):

        self.model = model
        self.X = X
        self.Y = Y
        self.N = N
        self.dt = dt
        self.normal_delta = normal_delta

        self.l_ax_x = l_ax_x
        self.l_ax_y = l_ax_y
        self.l_grid = l_grid
        self.n_points_grid = n_points_grid

        self.PICS_DIR_NAME = PICS_DIR_NAME

        self.is_draw_phi_dipol = is_draw_phi_dipol
        self.is_draw_pci_dipol = is_draw_pci_dipol
        self.is_draw_phi_ARCTG = is_draw_phi_ARCTG
        self.is_draw_pci_ARCTG = is_draw_pci_ARCTG
        self.is_draw_quiver = is_draw_quiver
        self.is_draw_quiver_in_points = is_draw_quiver_in_points

        self.show_pics = show_pics
        self.save_pics_and_params = save_pics_and_params


    def discrete_cmap(self, N, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""

        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)

    def draw_phi_dipol(self, X, Y, G, x0, y0, fig, ax):
        phi_val = f.phi_dipol(X, Y, G, x0, y0)
        min_val = np.min(phi_val)
        max_val = np.max(phi_val)

        color_num = 24
        if max_val < 0:
            colors1 = plt.cm.Reds(np.linspace(1, 0, color_num))
            colors2 = plt.cm.Blues(np.linspace(0, 1, 0))

        elif min_val < 0:
            l = (max_val - min_val) / color_num
            L = 0 - min_val
            k = L / l
            n = int(np.floor(k))
            m = color_num - n

            colors1 = plt.cm.Reds(np.linspace(1, 0, n))
            colors2 = plt.cm.Blues(np.linspace(0., 1, m))
        else:
            colors1 = plt.cm.Reds(np.linspace(1, 0, 0))
            colors2 = plt.cm.Blues(np.linspace(0., 1, color_num))

        colors = np.vstack((colors1, colors2))
        mymap_lin = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        mymap_discr = self.discrete_cmap(color_num, mymap_lin)


        ax.set_title('phi dipol')
        c = ax.pcolor(X, Y, phi_val, cmap=mymap_discr)
        fig.colorbar(c, ax=ax, orientation='horizontal')
        ax.scatter(x0, y0, s=0.5, c='red')

    def draw_pci_dipol(self, X, Y, G, x0, y0, fig, ax):
        pci_val = f.pci_dipol(X, Y, G, y0, x0)
        min_val = np.min(pci_val)
        max_val = np.max(pci_val)

        color_num = 24
        if min_val < 0:
            l = (max_val - min_val) / color_num
            L = 0 - min_val
            k = L / l
            n = int(np.floor(k))
            m = color_num - n

            colors1 = plt.cm.Reds(np.linspace(1, 0, n))
            colors2 = plt.cm.Blues(np.linspace(0., 1, m))
        else:

            colors1 = plt.cm.Reds(np.linspace(1, 0, 0))
            colors2 = plt.cm.Blues(np.linspace(0., 1, color_num))

        colors = np.vstack((colors1, colors2))
        mymap_lin = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        mymap_discr = self.discrete_cmap(color_num, mymap_lin)

        ax.set_title('pci dipol')
        c = ax.pcolor(X, Y, pci_val, cmap=mymap_discr)
        fig.colorbar(c, ax=ax)
        # ax.scatter(x0, y0, s=0.5, c='black')

    def draw_phi_ARCTG(self, X, Y, G, x0, y0, fig, ax):
        phi_val = f.phi_ARCTG(X, Y, G, x0, y0)
        min_val = np.min(phi_val)
        max_val = np.max(phi_val)

        color_num = 24
        if max_val < 0:
            colors1 = plt.cm.Reds(np.linspace(1, 0, color_num))
            colors2 = plt.cm.Blues(np.linspace(0, 1, 0))
        elif min_val < 0:
            l = (max_val - min_val) / color_num
            L = 0 - min_val
            k = L / l
            n = int(np.floor(k))
            m = color_num - n

            colors1 = plt.cm.Reds(np.linspace(1, 0, n))
            colors2 = plt.cm.Blues(np.linspace(0., 1, m))
        else:
            colors1 = plt.cm.Reds(np.linspace(1, 0, 0))
            colors2 = plt.cm.Blues(np.linspace(0., 1, color_num))

        colors = np.vstack((colors1, colors2))
        mymap_lin = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        mymap_discr = self.discrete_cmap(color_num, mymap_lin)

        # ax.set_title('phi ARCTG')
        ax.set_title(r'$\phi$')
        c = ax.pcolor(X, Y, phi_val, cmap=mymap_discr)
        # fig.colorbar(c, ax=ax, orientation='horizontal')
        fig.colorbar(c, ax=ax, shrink=0.3)
        ax.scatter(x0, y0, s=0.5, c='red')

    def draw_pci_ARCTG(self, X, Y, G, x0, y0, fig, ax):
        pci_val = f.pci_ARCTG(X, Y, G, x0, y0)
        min_val = np.min(pci_val)
        max_val = np.max(pci_val)

        color_num = 24
        if max_val < 0:
            colors1 = plt.cm.Reds(np.linspace(1, 0, color_num))
            colors2 = plt.cm.Blues(np.linspace(0, 1, 0))

        elif min_val < 0:
            l = (max_val - min_val) / color_num
            L = 0 - min_val
            k = L / l
            n = int(np.floor(k))
            m = color_num - n

            colors1 = plt.cm.Reds(np.linspace(1, 0, n))
            colors2 = plt.cm.Blues(np.linspace(0., 1, m))
        else:

            colors1 = plt.cm.Reds(np.linspace(1, 0, 0))
            colors2 = plt.cm.Blues(np.linspace(0., 1, color_num))

        colors = np.vstack((colors1, colors2))
        mymap_lin = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        mymap_discr = self.discrete_cmap(color_num, mymap_lin)

        # ax.set_title('pci ARCTG')
        # ax.set_title(r'$\psi$')
        c = ax.pcolor(X, Y, pci_val, cmap=mymap_discr)
        fig.colorbar(c, ax=ax, orientation='vertical')
        # fig.colorbar(c, ax=ax, shrink=0.3)
        ax.scatter(x0, y0, s=0.5, c='red')

    def draw_quiver(self, X, Y, G0, x0, y0, dt, ax):
        # X, Y = np.meshgrid(np.linspace(-1.05, -0.9, 300), np.linspace(-0.2, 0.2, 300))
        x_vec, y_vec = self.model.V(X, Y, G0, x0, y0, dt)
        x_vec = x_vec*0.1
        y_vec = y_vec*0.1
        # x_vec = x_vec
        # y_vec = y_vec
        skip = 5
        skip = 3
        # skip = 1
        # ax.scatter(x0, y0, s=3, c='red')
        ax.quiver(X[::skip,::skip], Y[::skip,::skip], x_vec[::skip,::skip], y_vec[::skip,::skip], scale_units='xy', scale=1.)

    def draw_quiver_in_one_point_with_normal(self, G0, x0, y0, x, y, ax, nxk, nyk):
        # normal delta
        delta = 0.01
        delta = 0.1
        # delta = 0
        # delta = 0.001

        u, v = 0, 0
        for j in range(0, len(G0)):
            u = u + self.model.velocity(G0[j], x0[j], y0[j], x + nxk*delta, y + nyk*delta)[0]
            v = v + self.model.velocity(G0[j], x0[j], y0[j], x + nxk*delta, y + nyk*delta)[1]

        ax.quiver(x, y, u*0.1, v*0.1, scale_units='xy', scale=1., width=0.005, headwidth=2) # вектор в точці вихора

    def draw_quiver_in_points(self, X, Y, G0, x0, y0, nxk, nyk, dt, ax):
        # normal delta
        # delta = 0.01
        # delta = 0.5
        # delta = 0.01
        # delta = 0
        # delta = 0.001
        delta = self.normal_delta
        # Ux і Uy вони потрібні для того, щоб не рахувати крайні точки в яких у нас немає нормалі
        Ux = x0
        Uy = y0

        x_vec = np.array([])
        y_vec= np.array([])

        for i in range(0, len(x0)):
            u, v = 0, 0
            for j in range(0, len(G0)):
                u = u + self.model.velocity(G0[j], x0[j], y0[j], x0[i] + nxk[i]*delta, y0[i] + nyk[i]*delta)[0]
                v = v + self.model.velocity(G0[j], x0[j], y0[j], x0[i] + nxk[i]*delta, y0[i] + nyk[i]*delta)[1]

            x_vec = np.append(x_vec, u)
            y_vec = np.append(y_vec, v)

        # ax.quiver(Ux, Uy, nxk*delta, nyk*delta, scale_units='xy', scale=1.) # нормаль
        ax.quiver(x0, y0, x_vec, y_vec, scale_units='xy', scale=1., width=0.001, headwidth=1) # вектор в точці вихора

    def init_cm_for_scatter(self, n):
        t_colormap = np.random.rand(n)
        # print(f't_colormap = {t_colormap}')
        # t_colormap = [0.2951, 0.9542, 0.3323, 0.0584, 0.3849, 0.3901, 0.6902, 0.8771, 0.0719, 0.7311,
        #               0.7538, 0.9521, 0.2363, 0.637,  0.7368, 0.9031, 0.1559, 0.1669, 0.5691, 0.3124,
        #               0.3913, 0.0506, 0.8485, 0.943,  0.8785, 0.2111, 0.7472, 0.6759, 0.8087, 0.6663,
        #               0.3707, 0.1582, 0.2641, 0.5866, 0.2553, 0.6612, 0.4369, 0.6401, 0.3195, 0.4388,
        #               0.5612, 0.5004, 0.9953, 0.2482, 0.8084, 0.2467, 0.9592, 0.0323, 0.7761, 0.0678,
        #               0.9686, 0.5126, 0.516,  0.0213, 0.7674, 0.71,   0.9358, 0.7092, 0.862,  0.4208,
        #               0.0142, 0.2992, 0.6148, 0.9088, 0.7051, 0.915,  0.1456, 0.1403, 0.7661, 0.7286,
        #               0.8452, 0.498,  0.324,  0.8425, 0.2777, 0.4325, 0.0011, 0.0435, 0.7784, 0.8943,
        #               0.8396, 0.9594, 0.9085, 0.3451, 0.7118, 0.2492, 0.44,   0.2971, 0.2589]
        def t():
            return t_colormap
        return t

    def _draw_info(self, G, x0, y0, x0_new, y0_new, nxk, nyk, fig, ax, t_cmap):
        if self.is_draw_phi_dipol:
            self.draw_phi_dipol(self.X, self.Y, G, x0, y0, fig, ax)
        elif self.is_draw_pci_dipol:
            self.draw_pci_dipol(self.X, self.Y, G, x0, y0, fig, ax)
        elif self.is_draw_phi_ARCTG:
            self.draw_phi_ARCTG(self.X, self.Y, G, x0, y0, fig, ax)
        elif self.is_draw_pci_ARCTG:
            self.draw_pci_ARCTG(self.X, self.Y, G, x0, y0, fig, ax)
        elif self.is_draw_quiver:
            self.draw_quiver(self.X, self.Y, G, x0_new, y0_new, self.dt, ax)
        elif self.is_draw_quiver_in_points:
            self.draw_quiver_in_points(self.X, self.Y, G, x0_new, y0_new, nxk, nyk, self.dt, ax)

        # t = t_cmap()
        # ax.scatter(x0_new, y0_new, s=3, c=t, cmap='hsv', zorder=2)
        ax.scatter(x0_new, y0_new, s=3, c='red' , zorder=2)

    def draw_surfaces(self, x0, y0, G):
        '''
        :param x0: vortex
        :param y0: vortex
        :param G:
        :param x: aproximatoin points of the line
        :param y: aproximatoin points of the line
        :param dt:
        :param N: number of lines to draw
        :return:
        '''
        t_cmap = self.init_cm_for_scatter(len(x0))

        fig = plt.figure()
        ax = plt.axes(xlim=(-self.l_ax_x, self.l_ax_x), ylim=(-self.l_ax_y, self.l_ax_y))
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'i = {1}')

        x0_new, y0_new, nxk, nyk = self.model.new_surface_2(x0, y0, G, x0, y0, self.dt)
        ax.plot(x0_new, y0_new)

        G = list(self.model.new_G(x0_new, y0_new, x0, y0, G, self.dt))

        self._draw_info(G, x0, y0, x0_new, y0_new, nxk, nyk, fig, ax, t_cmap)
        # self.draw_quiver_in_one_point_with_normal(G, x0_new, y0_new, x0_new[20], y0_new[20], ax, nxk[20], nyk[20])

        xk = (x0[1:] + x0[:-1]) / 2
        yk = (y0[1:] + y0[:-1]) / 2

        # ax.scatter(xk[75], yk[75], s=3, c='green', zorder=2)
        if self.save_pics_and_params:
            plt.savefig(f'{self.PICS_DIR_NAME}/pic{1}.png')
        if self.show_pics:
            plt.show()
        plt.clf()

        for i in tqdm(range(0, self.N-1)):
            ax = plt.axes(xlim=(-self.l_ax_x, self.l_ax_x), ylim=(-self.l_ax_y, self.l_ax_y))
            ax.set_aspect('equal', adjustable='box')

            # ось тут здається проблема з x0, y0 має бути x0_new, y0_new
            x0_new, y0_new, nxk, nyk = self.model.new_surface_2(x0, y0, G, x0, y0, self.dt)
            ax.plot(x0_new, y0_new)

            G = list(self.model.new_G(x0_new, y0_new, x0, y0, G, self.dt))

            ax.set_title(f'i = {i + 2}')
            self._draw_info(G, x0, y0, x0_new, y0_new, nxk, nyk, fig, ax, t_cmap)
            # self.draw_quiver_in_one_point_with_normal(G, x0_new, y0_new, x0_new[20], y0_new[20], ax, nxk[20], nyk[20])

            if self.save_pics_and_params:
                plt.savefig(f'{self.PICS_DIR_NAME}/pic{i+2}.png')
            if self.show_pics:
                    plt.show()
            plt.clf()

            x0 = x0_new
            y0 = y0_new