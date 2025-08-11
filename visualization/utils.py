import numpy as np


def gradient_line(
        x, y,
        ax,
        c,
        marker='o',
        scatter_size=15,
        lbl='', final_model_label=None,
        filter=None,
        starting_point=None,
        sample_ratio=1,
        threshold=None,
):
    if filter == 'x':
        filtered_x = [x[0]]
        filtered_y = [y[0]]
        lowest_x = x[0]
        for i in range(len(x)):
            if x[i] < lowest_x:
                lowest_x = x[i]
                filtered_x.append(lowest_x)
                filtered_y.append(y[i])
    elif filter == 'y':
        filtered_x = [x[0]]
        filtered_y = [y[0]]
        highest_y = y[0]
        for i in range(len(y)):
            if y[i] > highest_y:
                highest_y = y[i]
                filtered_x.append(x[i])
                filtered_y.append(highest_y)
    else:
        filtered_x = x
        filtered_y = y

    # Filter the points based on the sample ratio, keeping the first and last point
    if sample_ratio > 1:
        filtered_x = [filtered_x[0]] + filtered_x[1::sample_ratio] + [filtered_x[-1]]
        filtered_y = [filtered_y[0]] + filtered_y[1::sample_ratio] + [filtered_y[-1]]

    # Filter based on an absolute threshold change in either axis
    if threshold or threshold:
        # always take over the first point
        new_filtered_x = [filtered_x[0]]
        new_filtered_y = [filtered_y[0]]

        for i in range(1, len(filtered_x) - 1):
            distance = np.sqrt((filtered_x[i] - new_filtered_x[-1]) ** 2 + (filtered_y[i] - new_filtered_y[-1]) ** 2)
            if distance > threshold:
                new_filtered_x.append(x[i])
                new_filtered_y.append(y[i])

        # always take the last point
        new_filtered_x.append(filtered_x[-1])
        new_filtered_y.append(filtered_y[-1])

        filtered_x = new_filtered_x
        filtered_y = new_filtered_y

    n_points = len(filtered_x)
    alphas = np.linspace(0.30, 1, n_points)

    if starting_point is not None:
        # plot a dashed line connecting starting point to the first point of the filtered line
        ax.plot([starting_point[0], filtered_x[0]], [starting_point[1], filtered_y[0]], c=c, alpha=alphas[0],
                linestyle='--')

    for i, px, py, al in zip(range(len(filtered_x) - 1), filtered_x[:-1], filtered_y[:-1], alphas):
        ax.plot([px, filtered_x[i + 1]], [py, filtered_y[i + 1]], c=c, alpha=al)

    # Empty scatter for the legend entry
    ax.scatter([], [], c=c, alpha=1, label=lbl, marker=marker, s=scatter_size)
    ax.scatter(filtered_x, filtered_y, c=c, alpha=alphas, marker=marker, s=scatter_size)
    if final_model_label is not None:
        ax.annotate(
            final_model_label,
            xy=[filtered_x[-1], filtered_y[-1]],
            xytext=[filtered_x[-1] + 0, filtered_y[-1] - 0.001],
            fontsize=8,
        )

    return filtered_x, filtered_y


def get_label_position(x, y, x_prev, y_prev, offset_length=0.02):
    """
    Get the position of the label based on the previous trajectory.
    """

    # get direction between the two points
    dx = x - x_prev
    dy = y - y_prev

    # normalize the direction
    length = np.sqrt(dx ** 2 + dy ** 2)
    if length == 0:
        # If the points are the same, return a default offset
        return x + offset_length, y + offset_length
    dx /= length
    dy /= length
    # get the position of the label
    x_pos = x + dx * offset_length
    y_pos = y + dy * offset_length

    return x_pos, y_pos
