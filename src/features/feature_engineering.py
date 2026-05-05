import logging
import joblib
import pandas as pd
import dask.dataframe as dd

from pathlib import Path
from yaml import safe_load

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# create a logger object
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# attach to logger
logger.addHandler(console_handler)

def load_data(data_path:Path,usecols=["pickup_latitude","pickup_longitude"],chunksize=100000):

    # loading data in chunk
    df_iter = pd.read_csv(data_path,chunksize=chunksize,usecols=usecols)

    return df_iter

def fit_scaler(df_iter):

    scaler = StandardScaler()

    for chunk in df_iter:
        scaler.partial_fit(chunk)

    return scaler

def read_params(params_path:Path):

    with open(params_path,'r') as file:
        params = safe_load(file)

    return params

def save_model(model,save_path:Path):

    joblib.dump(model,save_path)

def fit_minibatch_kmeans(mini_batch_params,df_iter,scaler):

    mini_batch_kmeans = MiniBatchKMeans(**mini_batch_params)

    # train the kmean model
    for chunk in df_iter:
        # scaler transform
        scaler_chunk = scaler.transform(chunk)

        # mini batch kmeans partial fit
        mini_batch_kmeans.partial_fit(scaler_chunk)

    return mini_batch_kmeans

def main():

    # current path
    current_path = Path(__file__)

    # set root path
    root_path = current_path.parent.parent.parent

    # data path
    data_path = root_path/"data"/"interim"/"df_without_outliers.csv"

    # loading df_iter object
    df_iter = load_data(data_path)
    logger.info('data iter object loaded successfully')

    # fitting scaler object
    scaler = fit_scaler(df_iter)
    logger.info('scaler object fitted successfully')

    # scaler save path
    scaler_save_path = root_path/"models"/"scaler.joblib"

    # saving scaler object
    save_model(scaler,scaler_save_path)
    logger.info('scaler model saved successfully')

    # params path
    param_path = root_path/"params.yaml"

    # read the params
    params = read_params(params_path=param_path)
    mini_batch_params = params['feature_engineering']['mini_batch_kmeans']

    # loading df_iter object
    df_iter = load_data(data_path)
    logger.info('data iter object loaded successfully')

    # fit minibatch_kmeans object
    mini_batch_kmeans = fit_minibatch_kmeans(mini_batch_params,df_iter,scaler)
    logger.info('mini_batch_kmeans fitted successfully')

    # minibatch_kmeans save path
    mini_batch_kmeans_save_path = root_path/"models"/"mini_batch_kmeans.joblib"

    # save the mini batch kmeans model
    save_model(mini_batch_kmeans,mini_batch_kmeans_save_path)
    logger.info('mini_batch_kmeans model saved successfully')

    # read the data directly
    df = pd.read_csv(data_path,parse_dates=["tpep_pickup_datetime"])
    logger.info('Data read for cluster prediction')

    # creating location subset
    location_df = df[['pickup_longitude','pickup_latitude']]
    logger.info('location_df created')

    # scaling the location_df data
    scaled_location_df = scaler.transform(location_df)
    logger.info('location_df scaled')

    # get the cluster prediction
    cluster_prediction = mini_batch_kmeans.predict(scaled_location_df)
    logger.info('cluster_prediction done')
    
    # adding the cluster prediction column to the data
    df['region'] = cluster_prediction
    logger.info('region column added in original df')

    # drop the lat long columns from the data
    df.drop(columns=['pickup_latitude','pickup_longitude'],inplace=True)
    logger.info('lat long columns droped from original df')

    # set the datetime column as the index
    df.set_index('tpep_pickup_datetime',inplace=True)
    logger.info('tpep_pickup_datetime set as index')

    # create region_grp object
    region_grp = df.groupby('region')
    logger.info('region_grp created')

    # resample data in 15 mins intereval
    resampled_data = region_grp['region'].resample('15min').count()
    logger.info('resampled data generated')

    # assign the resample name
    resampled_data.name = 'total_pickups'

    # convert back to df
    resampled_data = resampled_data.reset_index(level=0)

    # replace zeros with aribitary epsilon value
    epsilon_val = 10
    resampled_data.replace({'total_pickups':{0:epsilon_val}},inplace=True)
    logger.info('0 total_pickups replaced by epsilon value')

    # load the ewma parameters
    ewma_params = params['feature_engineering']['ewma']

    # calculate avg_pickups
    resampled_data['avg_pickups'] = (
                resampled_data
                .groupby('region')['total_pickups']
                .ewm(**ewma_params)
                .mean()
                .shift(1)
                .round()
                .values
    )
    logger.info('Avg pickups calculated successfully using EWMA')

    # save the resampled data
    save_path = root_path/"data"/"processed"/"resampled_data.csv"
    resampled_data.to_csv(save_path,index=True)
    logger.info('Data saved successfully')


if __name__ == "__main__":
    main()

