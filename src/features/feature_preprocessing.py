import logging
import pandas as pd

from pathlib import Path

# create a logger object
logger = logging.getLogger('feature_preprocessing')
logger.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler
logger.addHandler(console_handler)

def load_data(data_path:Path,parse_dates=['tpep_pickup_datetime']):
    
    df = pd.read_csv(data_path,parse_dates=parse_dates)

    return df

def preprocessing_pipeline(df:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    
    # extract day of the week
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_of_week
    logger.info('day of the week extracted')

    # extract month
    df['month'] = df['tpep_pickup_datetime'].dt.month
    logger.info('month extracted')

    # set the datetime column as index
    df.set_index('tpep_pickup_datetime')

    # create the region group
    region_grp = df.groupby('region')
    logger.info('region group created')

    # shifting periods
    periods = list(range(1,5))

    # generate lag features
    lag_features = region_grp['total_pickups'].shift(periods)
    logger.info('lag features created')

    # merge with original df
    df = pd.concat([lag_features,df],axis=1)
    logger.info('lag features concatenated with original df')

    # drop the missing values 
    df.dropna(inplace=True)
    logger.info('null values dropped')

    # rename the columns
    mapper = {name:f"lag_{ind+1}" for ind,name in enumerate(df.columns[:4])}
    df = df.rename(columns=mapper)

    # creating train data
    train_df = df[df['month'].isin([1,2])]
    logger.info('train data created')

    # creating test data
    test_df = df[df['month'].isin([3])]
    logger.info('test data created')

    return train_df,test_df


def save_data(df:pd.DataFrame,save_path:Path)->None:
    
    df.to_csv(save_path,index=True)
    logger.info(f'data saved at {save_path}')

def main():

    # current path
    current_path = Path(__file__)

    # root path
    root_path = current_path.parent.parent.parent

    # data path
    data_path = root_path/"data"/"processed"/"resampled_data.csv"

    # load data
    df = load_data(data_path,parse_dates=['tpep_pickup_datetime'])
    logger.info('data loaded successfully')

    # data preprocessing
    train_df,test_df = preprocessing_pipeline(df)
    logger.info('train_df and test_df created successfully')

    # save train data path
    train_df_path = root_path/"data"/"processed"/"train_df.csv"

    # save test data path
    test_df_path = root_path/"data"/"processed"/"test_df.csv"

    # save the data
    save_data(train_df,train_df_path)
    logger.info('train data saved successfully')

    save_data(test_df,test_df_path)
    logger.info('test data saved successfully')

if __name__ == "__main__":
    main()