import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from moviepy.editor import *
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)

import FUNCTIONS as f
import LINE as line
import DRAW
from Models import Model_wNormal

normal_delta = 0.02
# normal_delta = 0
l_ax_x = 5
l_ax_y = 5
dt = 0.1
def main():
    def write_param(file_path, parameters):
        with open(file_path, 'a') as f:
            for key, value in parameters.items():
                if isinstance(value, np.ndarray):
                    f.write(f'{key} : ({value[0][0]}, {value[0][-1]}, {len(value[0])}),\n')
                else:
                    f.write(f'{key} : {value},\n')

            f.write(f'line_name : {line_function.__name__}')

    def show_line(show=False, save=False):
        l = 7.5
        ax = plt.axes(xlim=(-l, l), ylim=(-2.5, 2.5))
        ax.set_aspect('equal', adjustable='box')
        plt.plot(x0, y0, color='black')

        t = np.random.rand(len(x0))
        # ax.scatter(x0, y0, s=3, c=t, cmap='viridis', zorder=2)
        print(f'len(x0) = {len(x0)}')

        if save:
            plt.savefig(f'{parameters['PICS_DIR_NAME']}/pic{0}.png')
        if show:
            plt.show()

    def make_movie(PICS_DIR_NAME, duration=0.25):
        directory = PICS_DIR_NAME
        files = os.listdir(directory)
        images = list(filter(lambda x: x.endswith('.png'), files))

        images.sort(key=lambda x: int(x[3:-4]))

        # print(images)
        clips = [ImageClip(f'{PICS_DIR_NAME}/' + img).set_duration(duration) for img in images]
        final_clip = concatenate_videoclips(clips, method='compose')
        name = PICS_DIR_NAME.split('/')[-1]
        final_clip.write_videofile(f'{PICS_DIR_NAME}/{name}_video.mp4', fps=24)

    l_grid = 12
    n_points_grid = 100
    N = 1500
    model = Model_wNormal.Model_wNormal()

    X, Y = np.meshgrid(np.linspace(-l_grid, l_grid, n_points_grid), np.linspace(-l_grid, l_grid, n_points_grid))
    line_function = line.cosinusoid_line_longer
    line_function = line.complex_line_for_comp_hydromech_j
    # line_function = line.complex_line_for_comp_hydromech_j
    # line_function = line.complex_line_for_shev_vesna_j

    x0, y0, n_points = line_function()
    G = list(model.init_G(x0, y0))
    # PICS_DIR_NAME = f'pic/for_stud_konkurs/sinusoid/normal_delta=0.02/dt=0.01/cosinusoid_line_longer_with_normal_delta={normal_delta}_{n_points}_p_dt={dt}_{l_ax_y}x{l_ax_x}'
    # PICS_DIR_NAME = f'pic/experiment/cosinusoid_line_longer_with_normal_delta={normal_delta}_{n_points}_p_dt={dt}_{l_ax_y}x{l_ax_x}'

    PICS_DIR_NAME = f'pic/experiment/complex_line_for_comp_hydromech_j_with_normal_delta={normal_delta}_{n_points}_p_dt={dt}_{l_ax_y}x{l_ax_x}_ridge_ifelse=150'

    parameters = {
        'model': model,
        'X' : X,
        'Y' : Y,
        'N' : N,
        'dt' : dt,
        'normal_delta' : normal_delta,
        'l_ax_x' : l_ax_x,
        'l_ax_y' : l_ax_y,
        'l_grid' : l_grid,
        'n_points_grid' : n_points_grid,
        'PICS_DIR_NAME' : PICS_DIR_NAME,
        'is_draw_phi_dipol' : False,
        'is_draw_pci_dipol' : False,
        'is_draw_phi_ARCTG' : False,
        'is_draw_pci_ARCTG' : False,
        'is_draw_quiver' : False,
        'is_draw_quiver_in_points' : False,
        'show_pics' : False,
        'save_pics_and_params': True,
    }

    file_path = parameters['PICS_DIR_NAME']+'/parameters.txt'
    if parameters['save_pics_and_params']:
        if os.path.exists(file_path):
            os.remove(file_path)
        if not os.path.exists(parameters['PICS_DIR_NAME']):
            os.mkdir(parameters['PICS_DIR_NAME'])
        write_param(file_path, parameters)

    show_line(show=False, save=False)

    try:
        draw = DRAW.Draw(**parameters)
        draw.draw_surfaces(x0, y0, G)
        make_movie(parameters['PICS_DIR_NAME'], duration=0.1)
    except Exception as e:
        print(e)
        make_movie(parameters['PICS_DIR_NAME'], duration=0.25)

if __name__ == '__main__':
    main()
