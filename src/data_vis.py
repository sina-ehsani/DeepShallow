import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


def visualize_data7(input, taxi_zone_map, vmax=300):
    taxi_zone_map = taxi_zone_map.to_numpy()  # Convert to numpy array

    # set the size of the plot
    plt.figure(figsize=(9, 9))

    # create Color Map
    # define the colors and the positions of those colors
    colors = [[0, "honeydew"], [0.0001, "limegreen"], [0.5, "yellow"], [1, "orangered"]]
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

            if int(np.array(taxi_zone_map)[i, j]) == 0:
                # Change the color of the box to White
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color="lightblue")
                plt.gca().add_patch(rect)

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

    # add a color bar
    plt.colorbar()

    # show the plot
    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color="blue", label="Training loss")
    plt.plot(*zip(*val_losses), color="red", label="Validation loss")  # Unpack the tuples
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Over Time")
    plt.show()


def show_loss(model, val_pu_ts, val_do, val_labels, taxi_zone_map, device):
    with torch.no_grad():
        if val_pu_ts is not None:
            val_outputs = model(val_pu_ts.to(device), val_do.to(device))
        else:
            val_outputs = model(val_do.to(device))

    val_labels = val_labels.to("cpu")
    val_outputs = val_outputs.to("cpu")

    # Calculate the MSE of the validation outputs and validation data
    mse_loss = nn.MSELoss()
    val_mse = mse_loss(val_outputs, val_labels)

    # Calculate the MAE of the validation outputs and validation data
    mae_loss = nn.L1Loss()
    val_mae = mae_loss(val_outputs, val_labels)

    print(val_mse, val_mae)

    # np.array(taxi_zone_map)[9][3] #236
    # np.array(taxi_zone_map)[10][3] #237
    # np.array(taxi_zone_map)[11][3] #237

    rows, cols = np.where(taxi_zone_map == 237)
    val_mse_237 = mse_loss(val_outputs[:, :, rows, cols], val_labels[:, :, rows, cols])
    val_mae_237 = mae_loss(val_outputs[:, :, rows, cols], val_labels[:, :, rows, cols])

    print(f"validation MSE and MAE of block 237 (pixel) : {val_mse_237} , {val_mae_237} ")
    print(f"validation MSE and MAE of block 237 : {val_mse_237*len(rows)} , {val_mae_237*len(rows)} ")

    rows, cols = np.where(taxi_zone_map == 236)
    val_mse_236 = mse_loss(val_outputs[:, :, rows, cols], val_labels[:, :, rows, cols])
    val_mae_236 = mae_loss(val_outputs[:, :, rows, cols], val_labels[:, :, rows, cols])

    print(f"validation MSE and MAE of block 236 (pixel) : {val_mse_236} , {val_mae_236} ")
    print(f"validation MSE and MAE of block 236 : {val_mse_236*len(rows)} , {val_mae_236*len(rows)} ")

    print("Calculate the MAE of the validation outputs and validation data")
    mae_loss = nn.L1Loss(reduction="none")

    print(mae_loss(val_outputs, val_labels).shape)

    print(f"Maximum MSE: {torch.max(torch.mean(mae_loss(val_outputs, val_labels), (0,1) ))}")

    visualize_data7(torch.mean(mae_loss(val_outputs, val_labels), (0, 1)), taxi_zone_map, 10)


def create_predicted_real_df(model, val_pu_ts, val_do, val_labels, taxi_zone_map, device, date_list):
    with torch.no_grad():
        if val_pu_ts is not None:
            val_outputs = model(val_pu_ts.to(device), val_do.to(device))
        else:
            val_outputs = model(val_do.to(device))

    val_labels = val_labels.to("cpu")
    val_outputs = val_outputs.to("cpu")

    # Create a dataframe
    rows_list = []

    for i in range(val_labels.shape[0]):  # Assuming first dimension is the batch size
        for m in range(val_labels.shape[1]):
            date = date_list[i + m].date()  # Extracting just the date
            hour = date_list[i + m].hour  # Extracting the hour
            minute = date_list[i + m].minute  # Extracting the minute

            # We start each row with date, hour, and minute
            row = {"Date": date, "Hour": hour, "Minute Interval": minute}

            for j in range(val_labels.shape[-2]):  # Assuming second to last dimension corresponds to rows in your map
                for k in range(val_labels.shape[-1]):  # Assuming last dimension corresponds to cols in your map
                    location_id = taxi_zone_map.iloc[j, k]

                    true_value = val_labels[i, m, j, k].item()
                    predicted_value = val_outputs[i, m, j, k].item()
                    predicted_value = max(int(np.floor(predicted_value)), 0)

                    # Add true value and prediction for each location ID
                    row[f"{location_id}_gold"] = row.get(f"{location_id}_gold", 0) + true_value
                    row[f"{location_id}_pred"] = row.get(f"{location_id}_pred", 0) + predicted_value

            rows_list.append(row)

    df = pd.DataFrame(rows_list)

    return df
