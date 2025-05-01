import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_morphogen_profiles(folder_path):
    # Load data
    x_centerst = np.load(os.path.join(folder_path, 'x_centerst.npy'))
    st = np.load(os.path.join(folder_path, 'st.npy'))
    ft = np.load(os.path.join(folder_path, 'ft.npy'))
    time_to_save = np.load(os.path.join(folder_path, 'time_to_save.npy'))

    n_frames = x_centerst.shape[0]
    initial_idx = 0

    # Create figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    x = x_centerst[initial_idx]
    s_line, = ax.plot(x, st[initial_idx], label='s', lw=2)
    f_line, = ax.plot(x, ft[initial_idx], label='f', lw=2)

    ax.set_title(f"Time = {time_to_save[initial_idx]:.2f}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, max(np.max(st[initial_idx]), np.max(ft[initial_idx])) * 1.1)

    # Slider axis
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    time_slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=initial_idx, valstep=1)

    # Slider callback
    def update(val):
        idx = int(time_slider.val)
        x = x_centerst[idx]
        s_line.set_data(x, st[idx])
        f_line.set_data(x, ft[idx])
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, max(np.max(st[idx]), np.max(ft[idx])) * 1.1)
        ax.set_title(f"Time = {time_to_save[idx]:.2f}")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    plt.show()



plot_morphogen_profiles('growth_simulations/figure_s5_effect_of_growth/results_of_numerics/two_morph_gmultiplicative/dynamics/th_0.15_g_5.0')