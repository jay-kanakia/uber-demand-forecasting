import logging
import mlflow
import dagshub
import joblib
import pandas as pd

from pathlib import Path
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# create a logger object
logger = logging.getLogger('model_training')
logger.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# attach handler
logger.addHandler(console_handler)

def load_data(data_path:Path,parse_dates=["tpep_pickup_datetime"]):

    df = pd.read_csv(data_path,parse_dates=parse_dates)

    return df

def make_transformer(train_df:pd.DataFrame):
    
    # make X_test and y_test
    X_train = train_df.drop(columns=['total_pickups'])
    y_train = train_df['total_pickups']


    encoder = ColumnTransformer(
        [
            ('ohe',OneHotEncoder(drop='first',sparse_output=True),['region','day_of_week'])
        ],
        remainder='passthrough',verbose_feature_names_out=False
    )

    # fit encoder
    encoder.fit(X_train)

    return encoder

def model_training(train_df,encoder):
    
    # make X_test and y_test
    X_train = train_df.drop(columns=['total_pickups'])
    y_train = train_df['total_pickups']

    # transform X_train
    X_train_encoded = encoder.transform(X_train)

    # object creation
    lr = LinearRegression()

    # train the model
    lr.fit(X_train_encoded,y_train)

    return lr

def save_model(model,save_path:Path):

    joblib.dump(model,save_path)

def main():
    
    # current path
    current_path = Path(__file__)

    # root path
    root_path = current_path.parent.parent.parent

    # train data path
    train_df_path = root_path/"data"/"processed"/"train_df.csv"

    # load train data
    train_df = load_data(train_df_path)
    logger.info('train data loaded successfully')

    # setting datetime as index
    train_df.set_index('tpep_pickup_datetime',inplace=True)
    logger.info('train data index set')

    # make transformer encoder
    encoder = make_transformer(train_df)
    logger.info('encoder created successfully')

    # save encoder path
    save_encoder_path = root_path/"models"/"encoder.joblib"

    # save encoder
    save_model(encoder,save_encoder_path)
    logger.info('encoder saved successfully')

    # model training
    model = model_training(train_df,encoder)
    logger.info('model trained successfully')

    # save model path
    save_model_path = root_path/"models"/"model.joblib"

    # save model
    save_model(model,save_model_path)
    logger.info('model saved successfully')


if __name__ == "__main__":
    main()