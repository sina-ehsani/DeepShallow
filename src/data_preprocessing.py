import os
import random

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.autonotebook import tqdm


class TaxiData:
    def __init__(self, table, manhattan_zones):
        self.table = table
        # Create the bins for the intervals
        self.bins = [0, 15, 30, 45, 59]
        self.manhattan_zones = manhattan_zones

    def extract_pickup_days(self, pickup_mth):
        self.table_pu = self.table.copy()
        self.table_pu["pickup_day"] = self.table_pu["tpep_pickup_datetime"].dt.day
        self.table_pu["pickup_hour"] = self.table_pu["tpep_pickup_datetime"].dt.hour
        self.table_pu["pickup_yr_mth"] = self.table_pu["tpep_pickup_datetime"].dt.to_period("M")
        self.table_pu["DayofWeek"] = self.table_pu[
            "tpep_pickup_datetime"
        ].dt.dayofweek  # 0 for Monday, ... , Sunday = 6
        self.table_pu["DayofWeek_str"] = self.table_pu["tpep_pickup_datetime"].dt.day_name()

        # Add Min intervals:
        # Extract the minute from the 'time' column
        self.table_pu["pickup_min"] = self.table_pu["tpep_pickup_datetime"].dt.minute

        # Create the new column 'interval' using pd.cut()
        self.table_pu["pu_min_interval"] = pd.cut(
            self.table_pu["pickup_min"], self.bins, labels=False, include_lowest=True
        )

        filter = self.table_pu["pickup_yr_mth"] == pickup_mth  # filter by month
        self.table_pu = self.table_pu[filter]
        return self.table_pu

    def only_manhatan_pickup(self):
        filter2 = self.table_pu["PULocationID"].isin(self.manhattan_zones)
        self.table_pu = self.table_pu[filter2]
        return self.table_pu

    def filter_pu_weekdays(self):
        filter6 = self.table_pu["DayofWeek"] <= 4
        self.table_pu = self.table_pu[filter6]
        return self.table_pu

    def filter_pickup_hours(self, start, end):
        filter1 = self.table_pu["pickup_hour"] <= end - 1
        filter2 = start <= self.table_pu["pickup_hour"]
        self.table_pu = self.table_pu[filter1 & filter2]
        return self.table_pu

    def groupby_pickup_min_intervals(self):
        gp_cols = ["pickup_day", "pickup_hour", "pu_min_interval", "PULocationID"]
        agg_cols = {
            "pickup_yr_mth": "max",
            "passenger_count": "count",
            "total_amount": "sum",
            "DayofWeek": "max",
            "DayofWeek_str": "max",
        }
        return (
            self.table_pu.groupby(gp_cols)
            .agg(agg_cols)
            .sort_values(["pickup_day", "pickup_hour", "pu_min_interval", "PULocationID"])
            .reset_index()
        )

    def extract_dropoff_days(self, dropoff_mth):
        self.table_do = self.table.copy()
        self.table_do["dropoff_day"] = self.table_do["tpep_dropoff_datetime"].dt.day
        self.table_do["dropoff_hour"] = self.table_do["tpep_dropoff_datetime"].dt.hour
        self.table_do["dropoff_yr_mth"] = self.table_do["tpep_dropoff_datetime"].dt.to_period("M")
        self.table_do["DayofWeek"] = self.table_do[
            "tpep_dropoff_datetime"
        ].dt.dayofweek  # 0 for Monday, ... , Sunday = 6
        self.table_do["DayofWeek_str"] = self.table_do["tpep_dropoff_datetime"].dt.day_name()

        # Add Min intervals:
        # Extract the minute from the 'time' column
        self.table_do["dropoff_min"] = self.table_do["tpep_dropoff_datetime"].dt.minute

        # Create the new column 'interval' using pd.cut()
        self.table_do["do_min_interval"] = pd.cut(
            self.table_do["dropoff_min"], self.bins, labels=False, include_lowest=True
        )

        filter = self.table_do["dropoff_yr_mth"] == dropoff_mth  # filter by month
        self.table_do = self.table_do[filter]
        return self.table_do

    def only_manhatan_dropoff(self):
        filter2 = self.table_do["DOLocationID"].isin(self.manhattan_zones)
        self.table_do = self.table_do[filter2]
        return self.table_do

    def filter_dropoff_hours(self, start, end):
        filter1 = self.table_do["dropoff_hour"] <= end - 1
        filter2 = start <= self.table_do["dropoff_hour"]
        self.table_do = self.table_do[filter1 & filter2]
        return self.table_do

    def filter_do_weekdays(self):
        filter6 = self.table_do["DayofWeek"] <= 4
        self.table_do = self.table_do[filter6]
        return self.table_do

    def groupby_dropoff_min_intervals(self):
        gp_cols = ["dropoff_day", "dropoff_hour", "do_min_interval", "DOLocationID"]
        agg_cols = {
            "dropoff_yr_mth": "max",
            "passenger_count": "count",
            "total_amount": "sum",
            "DayofWeek": "max",
            "DayofWeek_str": "max",
        }
        return (
            self.table_do.groupby(gp_cols)
            .agg(agg_cols)
            .sort_values(["dropoff_day", "dropoff_hour", "do_min_interval", "DOLocationID"])
            .reset_index()
        )


def get_dropoff_pickup_table(table, manhattan_zones, month, do_start, do_end, pu_start, pu_end, only_weekdays=False):
    """
    Create a table with pickup and dropoff data for a given month and time interval.

    Parameters:
      - table (pandas DataFrame): The table with the taxi data
      - manhattan_zones (numpy array): A list of the taxi zones in Manhattan
      - month (str): The month to filter by
      - do_start (int): The starting hour for the dropoff time interval
      - do_end (int): The ending hour for the dropoff time interval
      - pu_start (int): The starting hour for the pickup time interval
      - pu_end (int): The ending hour for the pickup time interval

    Returns:
      - pickup_table (pandas DataFrame): The table with the pickup data
      - dropoff_table (pandas DataFrame): The table with the dropoff data
    """
    # Create an instance of the DataPreparation class
    data_prep = TaxiData(table, manhattan_zones)

    # Extract the pickup data
    data_prep.extract_pickup_days(month)

    # Filter the pickup data by the pickup time interval
    data_prep.filter_pickup_hours(pu_start, pu_end)

    # Only Manhattan:
    data_prep.only_manhatan_pickup()

    # Filter the pickup data by weekdays
    if only_weekdays:
        data_prep.filter_pu_weekdays()

    # Group the pickup data by day, hour, and 15 min intervals
    pickup_table = data_prep.groupby_pickup_min_intervals()

    # Extract the dropoff data
    data_prep.extract_dropoff_days(month)

    # Filter the dropoff data by the dropoff time interval
    data_prep.filter_dropoff_hours(do_start, do_end)

    # Only Manhattan:
    data_prep.only_manhatan_dropoff()

    # Filter the dropoff data by weekdays
    if only_weekdays:
        data_prep.filter_do_weekdays()

    # Group the dropoff data by day, hour, and 15 min intervals
    dropoff_table = data_prep.groupby_dropoff_min_intervals()

    return pickup_table, dropoff_table


class DataTensorCreation:
    def __init__(self, taxi_zone_map):
        self.taxi_zone_map = taxi_zone_map

    def create_tensor(self, grouped_table):
        """
        Create a tensor from a grouped table and a taxi zone map.

        Parameters:
          - grouped_table (pandas DataFrame): The table that has been grouped by day, hour, and 15 min intervals
          - taxi_zone_map (numpy array): A map of the taxi zones, where each zone is represented by a unique integer

        Returns:
          - tensor (numpy array): A 3D tensor, where the first dimension represents the day,
          the second dimension represents the hour and 15 min intervals,
          and the third dimension represents the taxi zones and the passenger count in each zone.
        """
        all_data = []

        for month in grouped_table.pickup_yr_mth.unique():
            mnth_grouped_table = grouped_table[grouped_table["pickup_yr_mth"] == month]
            for day in mnth_grouped_table.pickup_day.unique():
                day_grouped_table = mnth_grouped_table[mnth_grouped_table["pickup_day"] == day]

                timeseries_traffic_maps = []
                for hour in day_grouped_table.pickup_hour.unique():
                    for min_interval in day_grouped_table.pu_min_interval.unique():
                        filterday2 = day_grouped_table["pickup_hour"] == hour
                        filterday3 = day_grouped_table["pu_min_interval"] == min_interval
                        day_grouped_table_h_m = day_grouped_table[filterday2 & filterday3]

                        taxi_zone_traffic_map = np.zeros_like(self.taxi_zone_map)

                        for row in day_grouped_table_h_m.itertuples():
                            taxi_zone_traffic_map[self.taxi_zone_map == row.PULocationID] = row.passenger_count

                        timeseries_traffic_maps.append(taxi_zone_traffic_map)
                all_data.append(np.stack(timeseries_traffic_maps))
        self.tensor = np.stack(all_data).astype(int)
        return self.tensor

    def time_series_tensor(self, window_size=5):
        """
        This function generates time series tensors by dividing the input tensor into windows of a specified size and randomly masking a portion of the last data point in each window.

        Parameters:
        window_size (int, optional): The size of the windows to divide the input tensor into. Default is 5.

        Returns:
        tuple: A tuple containing two numpy arrays, the first being the stack of all the windows, and the second being the stack of the last data point in each window with a portion of it masked.
        """
        all_time_series = []
        output_tensor = []
        tensor = self.tensor

        for i in range(window_size, len(tensor) + 1):
            all_time_series.append(np.copy(tensor[i - window_size : i]))
            output_tensor.append(np.copy(tensor[i - 1, 8:]))
            random_masking = random.randint(8, 14)
            all_time_series[-1][-1, random_masking:] = -1

        output_tensor = np.stack(output_tensor)
        all_time_series = np.stack(all_time_series)

        return all_time_series, output_tensor

    def mask_each_interval(self):
        """
        This function only predicts one time interval at a time, so it masks all the time intervals after the one being predicted.

        Returns:
        tuple: A tuple containing two numpy arrays, the first being the stack of all the windows, and the is a 2d tensor of the time interval that we want to predict.
            input_tensor: size (Day_interval (Day x intervals (7) ), time_intervals (8+7) , 2D tensor of zones (27 x 8) )
            output_tensor: size (Day_interval (Day x intervals (7) ) , 2D tensor of zones (27 x 8) -> Only outputs one interval per each input tensor)
        """
        input_tensor = []
        output_tensor = []
        tensor = self.tensor

        for i in range(1, len(tensor) + 1):
            for j in range(8, 16):
                input_tensor.append(np.copy(tensor[i - 1]))  # Copy the whole tensor
                output_tensor.append(np.copy(tensor[i - 1, j]))  # Copy the time interval that we want to predict
                input_tensor[-1][j:] = -1  # Mask all the time intervals after the one being predicted
                input_tensor[-1] = np.delete(
                    input_tensor[-1], -1, axis=0
                )  # Remove the last interval from the input tensor

        output_tensor = np.stack(output_tensor)
        input_tensor = np.stack(input_tensor)

        return input_tensor, output_tensor


class DataTensorCreationDropoffandPickup:
    def __init__(self, taxi_zone_map):
        self.taxi_zone_map = taxi_zone_map
        # Calculate the number of pixels for each zone
        self.zone_pixel_count = {
            zone: np.sum(np.sum(self.taxi_zone_map == zone)) for zone in np.unique(self.taxi_zone_map)
        }

    def create_tensor(self, grouped_pickup, grouped_dropoff, use_avg=False):
        """Create a tensor from a grouped table and a taxi zone map.

        Args:
            grouped_pickup (pd.DataFrame): The Pick-up that has been grouped by day, hour, and 15 min intervals
            grouped_dropoff (pd.DataFrame): The Drop-off that has been grouped by day, hour, and 15 min intervals
        """
        pickup_data = []
        dropoff_data = []

        for month in grouped_pickup.pickup_yr_mth.unique():
            month_pickup = grouped_pickup[grouped_pickup["pickup_yr_mth"] == month]
            month_dropoff = grouped_dropoff[grouped_dropoff["dropoff_yr_mth"] == month]
            for day in month_pickup.pickup_day.unique():
                day_pickup = month_pickup[month_pickup["pickup_day"] == day]
                day_dropoff = month_dropoff[month_dropoff["dropoff_day"] == day]

                timeseries_pu_maps = []
                timeseries_do_maps = []
                for hour in day_pickup.pickup_hour.unique():
                    for min_interval in day_pickup.pu_min_interval.unique():
                        filterday2 = day_pickup["pickup_hour"] == hour
                        filterday3 = day_pickup["pu_min_interval"] == min_interval
                        day_pickup_h_m = day_pickup[filterday2 & filterday3]
                        taxi_zone_pu_map = np.zeros_like(self.taxi_zone_map)
                        for row in day_pickup_h_m.itertuples():
                            if use_avg:
                                avg_count = row.passenger_count / self.zone_pixel_count[row.PULocationID]
                                taxi_zone_pu_map[self.taxi_zone_map == row.PULocationID] = avg_count
                            else:
                                taxi_zone_pu_map[self.taxi_zone_map == row.PULocationID] = row.passenger_count
                        timeseries_pu_maps.append(taxi_zone_pu_map)
                pickup_data.append(np.stack(timeseries_pu_maps))

                for hour in day_dropoff.dropoff_hour.unique():
                    for min_interval in day_dropoff.do_min_interval.unique():
                        filterday4 = day_dropoff["dropoff_hour"] == hour
                        filterday5 = day_dropoff["do_min_interval"] == min_interval
                        day_dropoff_h_m = day_dropoff[filterday4 & filterday5]
                        taxi_zone_do_map = np.zeros_like(self.taxi_zone_map)
                        for row in day_dropoff_h_m.itertuples():
                            if use_avg:
                                avg_count = row.passenger_count / self.zone_pixel_count[row.DOLocationID]
                                taxi_zone_do_map[self.taxi_zone_map == row.DOLocationID] = avg_count
                            else:
                                taxi_zone_do_map[self.taxi_zone_map == row.DOLocationID] = row.passenger_count
                        timeseries_do_maps.append(taxi_zone_do_map)
                dropoff_data.append(np.stack(timeseries_do_maps))

        self.pickup_data_tensor = np.stack(pickup_data).astype(int)
        self.dropoff_data_tensor = np.stack(dropoff_data).astype(int)

        # To round up:
        # self.pickup_data_tensor  = np.ceil(np.stack(pickup_data)).astype(int)
        # self.dropoff_data_tensor  = np.ceil(np.stack(dropoff_data)).astype(int)

        return self.pickup_data_tensor, self.dropoff_data_tensor

    def time_series_tensor(self, window_size=5):
        """
        This function generates time series tensors by dividing the input tensor into windows of a specified size.

        Parameters:
        window_size (int, optional): The size of the windows to divide the input tensor into. Default is 5.

        Returns:
        all_time_series (numpy array): A 3D tensor, where the first dimension represents the day,
        output_tensor (numpy array): The last data point in each window.
        """
        all_time_series = []
        tensor = self.pickup_data_tensor

        for i in range(window_size, len(tensor)):
            all_time_series.append(np.copy(tensor[i - window_size : i]))

        all_time_series = np.stack(all_time_series)

        self.pickup_data_tensor = self.pickup_data_tensor[window_size:]
        self.dropoff_data_tensor = self.dropoff_data_tensor[window_size:]

        return all_time_series, self.pickup_data_tensor, self.dropoff_data_tensor


def read_and_create_pickup_dropoff(
    table_location, manhattan_zones, do_start, do_end, pu_start, pu_end, only_weekdays=False
):
    pickup_df = []
    dropoff_df = []

    for table_name in tqdm(os.listdir(table_location), desc="Epoch"):
        # for table_name in os.listdir(table_location):
        table = pq.read_table(os.path.join(table_location, table_name))
        table = table.to_pandas()
        month = table_name.split("_")[-1].split(".")[0]
        table_pickup, table_dropoff = get_dropoff_pickup_table(
            table, manhattan_zones, month, do_start, do_end, pu_start, pu_end, only_weekdays
        )
        pickup_df.append(table_pickup)
        dropoff_df.append(table_dropoff)

    pickup_df = pd.concat(pickup_df)
    dropoff_df = pd.concat(dropoff_df)

    pickup_df = pickup_df.sort_values(
        ["pickup_yr_mth", "pickup_day", "pickup_hour", "PULocationID", "pu_min_interval"]
    ).reset_index(drop=True)
    dropoff_df = dropoff_df.sort_values(
        ["dropoff_yr_mth", "dropoff_day", "dropoff_hour", "DOLocationID", "do_min_interval"]
    ).reset_index(drop=True)

    return pickup_df, dropoff_df


def main(manhattan_zones, taxi_zone_map, do_start, do_end, pu_start, pu_end, only_weekdays=False, window_size=7):
    # do_start = 6
    # do_end = 10
    # pu_start = 16
    # pu_end = 20
    # only_weekdays= False

    # 2023 Data:
    test_location = "/content/drive/MyDrive/Data/Data_Taxi/Yellow_Cap_NYC_2023_Test"
    # 2021 and 2022 Data:
    train_location = "/content/drive/MyDrive/Data/Data_Taxi/Yellow_Cap_NYC_Train"

    test_pickup, test_dropoff = read_and_create_pickup_dropoff(
        test_location, manhattan_zones, do_start, do_end, pu_start, pu_end, only_weekdays
    )
    train_pickup, train_dropoff = read_and_create_pickup_dropoff(
        train_location, manhattan_zones, do_start, do_end, pu_start, pu_end, only_weekdays
    )

    # data_tensor_creation = DataTensorCreation( taxi_zone_map)

    # tensor = data_tensor_creation.create_tensor(train_pickup)
    # tensor_ts , tensor_ts_out = data_tensor_creation.time_series_tensor(window_size=5)
    # tensor_one_inter_train , tensor_one_inter_train_out = data_tensor_creation.mask_each_interval()

    # tensor_test = data_tensor_creation.create_tensor(test_pickup)
    # tensor_ts_test , tensor_ts_test_out = data_tensor_creation.time_series_tensor(window_size=5)
    # tensor_one_inter_test , tensor_one_inter_test_out = data_tensor_creation.mask_each_interval()

    # print(tensor.shape, tensor_test.shape)
    # print(tensor_one_inter_train.shape,tensor_one_inter_test_out.shape)
    # print(np.array_equal(tensor_one_inter_test[-1][-1] , tensor_one_inter_test_out[-2]))

    # When you want pickup and dropoff data:
    # window_size =7
    data_tensor_creation = DataTensorCreationDropoffandPickup(taxi_zone_map)

    tensor_pu, tensor_do = data_tensor_creation.create_tensor(train_pickup, train_dropoff, use_avg=True)
    tensor_pu_ts, tensor_pu, tensor_do = data_tensor_creation.time_series_tensor(window_size=window_size)

    # tensor_test_pu , tensor_test_do = data_tensor_creation.create_tensor(test_pickup , test_dropoff)
    tensor_test_pu, tensor_test_do = data_tensor_creation.create_tensor(test_pickup, test_dropoff, use_avg=True)
    tensor_test_pu_ts, tensor_test_pu, tensor_test_do = data_tensor_creation.time_series_tensor(window_size=window_size)

    # Checks:
    print(tensor_pu.shape, tensor_do.shape, tensor_pu_ts.shape)
    print(tensor_test_pu_ts.shape, tensor_test_pu.shape, tensor_test_do.shape)
    print(np.array_equal(tensor_test_pu[0], tensor_test_pu_ts[1][-1]))

    # Save the tensors:
    with h5py.File(f"/content/drive/MyDrive/Data/Data_Taxi/all_tensors_{window_size}_zone2.h5", "w") as hf:
        hf.create_dataset("tensor_pu_ts", data=tensor_pu_ts)
        hf.create_dataset("tensor_pu", data=tensor_pu)
        hf.create_dataset("tensor_do", data=tensor_do)
        hf.create_dataset("tensor_test_pu_ts", data=tensor_test_pu_ts)
        hf.create_dataset("tensor_test_pu", data=tensor_test_pu)
        hf.create_dataset("tensor_test_do", data=tensor_test_do)
