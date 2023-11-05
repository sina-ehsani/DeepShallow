import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def visualize_data(input, taxi_zone_map, vmax=300):
    # set the size of the plot
    plt.figure(figsize=(9, 9))

    # create Color Map
    # define the colors and the positions of those colors
    colors = [[0, "lightgray"], [0.02, "limegreen"], [0.6, "yellow"], [1, "orangered"]]
    cmap_name = "green_yellow_red"
    green_yellow_red = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    # set the min and max values for the color scale
    vmin = 0
    vmax = vmax

    # create the heat map
    plt.imshow(input, cmap=green_yellow_red, vmin=vmin, vmax=vmax)

    # remove x and y axis labels
    plt.xticks([])
    plt.yticks([])

    # loop through the matrix and add names to the values
    for i in range(taxi_zone_map.shape[0]):
        for j in range(taxi_zone_map.shape[1]):
            if int(np.array(taxi_zone_map)[i, j]) != 0:
                plt.text(j, i, int(np.array(taxi_zone_map)[i, j]), ha="center", va="center", color="black", fontsize=6)
            else:
                # Change the color of the box to White
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color="white")
                plt.gca().add_patch(rect)

    # add a color bar
    plt.colorbar()

    # show the plot
    plt.show()


def visualize_data6(taxi_zone_map_df):
    taxi_zone_map = taxi_zone_map_df.to_numpy()  # Convert to numpy array

    # Create a placeholder image with all zones in green (assuming our colormap spans from 0-1, and mid green is at 0.5 for example)
    display_map = np.ones_like(taxi_zone_map) * 0.5

    # Update zones with value 0 to light blue color value (assuming our colormap spans from 0-1, and light blue is at 0 for example)
    display_map[taxi_zone_map == 0] = 0

    # set the size of the plot
    plt.figure(figsize=(9, 9))

    # Define a green colormap: from light blue (0) to light green (0.5) to dark green (1)
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", ["lightblue", "darkgray", "green"])

    # Display the map using the green colormap
    plt.imshow(display_map, cmap=cmap, vmin=0, vmax=1)

    # remove x and y axis labels
    plt.xticks([])
    plt.yticks([])

    rows, cols = taxi_zone_map.shape

    # Check for boundary changes and draw lines
    for i in range(rows):
        for j in range(cols):
            # Check right neighbor
            if j < cols - 1 and taxi_zone_map[i, j] != taxi_zone_map[i, j + 1]:
                plt.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color="gray")
            # Add line at the right edge of the map
            elif j == cols - 1:
                plt.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color="gray")

            # Check neighbor below
            if i < rows - 1 and taxi_zone_map[i, j] != taxi_zone_map[i + 1, j]:
                plt.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color="gray")
            # Add line at the bottom edge of the map
            elif i == rows - 1:
                plt.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color="gray")

    # add zone numbers
    unique_zones = np.unique(taxi_zone_map)
    for zone in unique_zones:
        # Exclude the 0 zone
        if zone == 0:
            continue
        # get the coordinates of all cells in this zone
        y, x = np.where(taxi_zone_map == zone)
        # compute the centroid
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        # add the label
        plt.text(
            centroid_x,
            centroid_y,
            str(int(zone)),
            ha="center",
            va="center",
            color="black",
            fontsize=6,
            weight="bold",
            rotation=30,
        )

    # show the plot
    plt.tight_layout()
    plt.show()
