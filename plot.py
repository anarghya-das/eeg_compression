import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def pareto_line(sim, x_idx=0, y_idx=1):
    # sort sim by cr
    sim = sim[sim[:, x_idx].argsort()]
    lows = []
    opt = 0
    for i in range(len(sim)):
        if sim[i][x_idx] >= sim[opt][x_idx] and sim[i][y_idx] <= sim[opt][y_idx]:
            lows.append(sim[i])
            opt = i
    return np.array(lows, dtype=object)


def update(val, np_simulation, cr_slider, prd_slider, scatter, fig):
    indices = np.where(np.logical_and(
        np_simulation[:, 0] <= cr_slider.val, np_simulation[:, 1] <= prd_slider.val))
    scatter.set_offsets(np_simulation[indices[0], 0:2])
    fig.canvas.draw_idle()


def hover(event, np_simulation, annot, ax, scatter, fig):
    visible = annot.get_visible()
    if event.inaxes == ax:
        is_contained, annotation_index = scatter.contains(event)
        if is_contained:
            data_index = annotation_index["ind"][0]
            data_point_location = scatter.get_offsets()[data_index]
            annot.xy = data_point_location
            text_label = f"w: {np_simulation[:, 2][data_index]}, l: {np_simulation[:, 3][data_index]}, t:{np_simulation[:,4][data_index]:.4f}%, cr: {1/data_point_location[0]:.2f}, prd:{np_simulation[:,1][data_index]: .2f}%, e: {np_simulation[:,6][data_index]*100: .2f}%"
            annot.set_text(text_label)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if visible:
                annot.set_visible(False)
                fig.canvas.draw_idle()


def plot_simulation_results(np_simulation=None, file=None, title=None):
    pareto = None
    if np_simulation is None:
        np_simulation = np.load(f"{file}.npy", allow_pickle=True)
        # pareto = np.load(f"{file}_pareto.npy", allow_pickle=True)

    np_simulation[:, 6] = 1-np_simulation[:, 6].astype(float)
    pareto = pareto_line(np_simulation, y_idx=6)
    # accuracy = outer_line(np_simulation, y_idx=6)
    fig, ax = plt.subplots()
    if title:
        plt.title(title)
    else:
        plt.title(f"Simulation Results ({len(pareto)})")
    plt.xlabel("CR")
    plt.ylabel("Model Error (%)")
    fig.subplots_adjust(left=0.20, bottom=0.20)
    scatter = plt.scatter(
        np_simulation[:, 0], np_simulation[:, 1])
    plt.scatter(pareto[:, 0], pareto[:, 6]*100,
                marker='^', color='red')
    # matplotlib triangle markers
    # # https://matplotlib.org/3.1.1/api/markers_api.html
    # plt.plot(pareto[:, 0], pareto[:, 6]*100,
    #          marker='^', markersize=10, color='red')
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    ax_prd = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    ax_cr = fig.add_axes([0.25, 0.1, 0.65, 0.03])

    cr_slider = Slider(
        ax=ax_cr,
        label='CR',
        valmin=0.1,
        valmax=1,
        valinit=1,
    )

    prd_slider = Slider(
        ax=ax_prd,
        label="PRD (%)",
        valmin=0,
        valmax=100,
        valinit=100,
        orientation="vertical"
    )

    # prd_slider.on_changed(lambda event: update(
    #     event, np_simulation, cr_slider, prd_slider, scatter, fig))
    # cr_slider.on_changed(lambda event: update(
    #     event, np_simulation, cr_slider, prd_slider, scatter, fig))
    fig.canvas.mpl_connect("motion_notify_event",
                           lambda event: hover(event, np_simulation, annot, ax, scatter, fig))
    # plt.figure(2)
    # # const = np.where(np.logical_and(
    # # pareto[:, 0] <= 0.6, pareto[:, 1] <= 10))[0]
    # labels, values = np.unique(pareto[:, 2], return_counts=True)
    # plt.bar(labels, values)
    plt.show()


plot_simulation_results(file="wa_pca_comb")
