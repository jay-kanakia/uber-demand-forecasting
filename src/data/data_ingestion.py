import logging
import dask.dataframe as dd
from pathlib import Path

# create a logger
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.INFO)

# create a handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler
logger.addHandler(console_handler)

# inlier range for latitude and longitude
min_latitude = 40.60
max_latitude = 40.85
min_longitude = -74.05
max_longitude = -73.70

# inlier range for fare amount and trip distance
min_fare_amount_val = 0.50
max_fare_amount_val = 81.0
min_trip_distance_val = 0.25
max_trip_distance_val = 24.43

def read_dask_df(data_path:Path, parse_dates:list=["tpep_pickup_datetime"],columns:list=['trip_distance', 
                                'tpep_pickup_datetime', 
                                'pickup_longitude',
                                'pickup_latitude',
                                'dropoff_longitude', 
                                'dropoff_latitude', 
                                'fare_amount']):
    
    dd_df = dd.read_csv(data_path,usecols=columns,parse_dates=parse_dates)

    return dd_df

def dask_pipeline(df:dd):
    
    # filtering lat and long
    pickup_latitude_bool = df['pickup_latitude'].between(min_latitude,max_latitude,inclusive='both')
    pickup_longitude_bool = df['pickup_longitude'].between(min_longitude,max_longitude,inclusive='both')
    dropoff_latitude_bool = df['dropoff_latitude'].between(min_latitude,max_latitude,inclusive='both')
    dropoff_longitude_bool = df['dropoff_longitude'].between(min_longitude,max_longitude,inclusive='both')

    # removing lat long outlier
    df = df[pickup_latitude_bool & pickup_longitude_bool & dropoff_latitude_bool & dropoff_longitude_bool]
    logger.info('lat long outlier removed')

    # filtering distance and fare amount
    fare_amount_bool = df['fare_amount'].between(min_fare_amount_val,max_fare_amount_val,inclusive='both')
    trip_distance_bool = df['trip_distance'].between(min_trip_distance_val,max_trip_distance_val,inclusive='both')

    # removing distance and fare amount outlier
    df = df[fare_amount_bool & trip_distance_bool]
    logger.info('trip distance and fare amount outlier removed')

    # compute df
    df = df.compute()
    logger.info('Dask dataframe is computed successfully')

    return df

def main():
    
    # current path
    current_path = Path(__file__)

    #root path
    root_path = current_path.parent.parent.parent

    # raw data path
    raw_data_dir = root_path/"data"/"raw"

    # dataframes names
    df_names = ["yellow_tripdata_2016-01.csv","yellow_tripdata_2016-02.csv","yellow_tripdata_2016-03.csv"]

    #read all dataframes
    dfs = []

    # loop
    for df_name in df_names:
        df_path = raw_data_dir/df_name
        df = read_dask_df(df_path)
        dfs.append(df)
    logger.info('Dask DataFrames are read successfully')  

    # concatenate all dfs
    df_final = dd.concat(dfs,axis=0)
    logger.info("All datasets merged successfully")

    # execute the dask pipeline
    df_final = dask_pipeline(df_final)
    logger.info("Dask pipeline is executed successfully")

    # save the dataframe
    df_without_outlier_path = root_path/"data"/"interim"/"df_without_outliers.csv"
    df_final.to_csv(df_without_outlier_path,index=False)
    logger.info("Dataframe is saved successfully")

if __name__ == "__main__":
    main()