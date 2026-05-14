import logging
import dagshub
import mlflow
import joblib
import json

import pandas as pd

from pathlib import Path
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error

# create a logger
logger = logging.getLogger('Model evaluation and registration')
logger.setLevel('INFO')

# create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('INFO')

#create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# attach logger
logger.addHandler(console_handler)

# initialize dagashub
dagshub.init(repo_owner='jay-kanakia', repo_name='uber-demand-forecasting', mlflow=True)

# set mlflow tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/uber-demand-forecasting.mlflow')

# set mlflow experiment name
mlflow.set_experiment('DVC Pipeline')

def load_data(data_path:Path,parse_dates=['tpep_pickup_datetime']):

    df = pd.read_csv(data_path,parse_dates=parse_dates)

    return df

def load(model_path:Path):
    model = joblib.load(model_path)
    return model

def data_pipeline(test_df,encoder,model):

    test_df.set_index('tpep_pickup_datetime',inplace=True)

    # Make X_test and y_test
    X_test = test_df.drop(columns=['total_pickups'])
    y_test = test_df['total_pickups']
    logger.info('test data splitted into X_test and y_test successfully')

    # transform the test data
    X_test_encoded = encoder.transform(X_test)
    logger.info('X_test data encoded successfully')

    # make prediction
    y_pred = model.predict(X_test_encoded)
    logger.info('y_pred calculated successfully')

    # calculate the loss
    loss = mean_absolute_percentage_error(y_test,y_pred)
    logger.info('loss calculated successfully')

    return (X_test_encoded,y_pred,loss)

def save_run_information(run_id,artifact_path,model_uri,save_path):
    run_info = {
        'run_id' : run_id,
        'artifact_path' : artifact_path,
        'model_uri' : model_uri,
    }

    with open(save_path,'w') as file:
        json.dump(run_info,file,indent=4)

def main():

    # current path
    current_dir = Path(__file__)

    # root path
    root_path = current_dir.parent.parent.parent

    # train data path
    train_data_path = root_path/"data"/"processed"/"train_df.csv"

    # load test data
    train_df = load_data(train_data_path)
    logger.info('train data loaded successfully')

    # test data path
    test_data_path = root_path/"data"/"processed"/"test_df.csv"

    # load test data
    test_df = load_data(test_data_path)
    logger.info('test data loaded successfully')

    # encoder path
    encoder_path = root_path/"models"/"encoder.joblib"

    # load encoder
    encoder = load(encoder_path)
    logger.info('encoder loaded successfully')

    # model path
    model_path = root_path/"models"/"model.joblib"

    # load model
    model = load(model_path)
    logger.info('model loaded successfully')

    # data pipeline
    X_test_encoded,y_pred,loss = data_pipeline(test_df,encoder,model)

    # mlflow tracking
    with mlflow.start_run(run_name='model'):

        # log model parameters
        mlflow.log_params(model.get_params())
        logger.info('model params logged successfully')

        # log the metric
        mlflow.log_metric('MAPE',loss)
        logger.info('MAPE logged successfully')

        # convert pandas dataset into mlflow datasets
        training_data = mlflow.data.from_pandas(train_df,targets='total_pickups')
        testing_data = mlflow.data.from_pandas(test_df,targets='total_pickups')

        # log the datasets
        mlflow.log_input(training_data,'training')
        logger.info('train dataset logged successfully')
        
        mlflow.log_input(testing_data,'validation')
        logger.info('test dataset logged successfully')

        # model signature
        model_signature = mlflow.models.infer_signature(X_test_encoded,y_pred)
        logger.info('model signature logged successfully')

        # log sklearn model
        logged_model = mlflow.sklearn.log_model(model,'demand_prediction',signature=model_signature,pip_requirements='requirements.txt',registered_model_name='uber_demand_prediction')
        logger.info('model logged successfully')

    # get run_id, artifact_path,model_uri
    run_id = logged_model.run_id
    artifact_path = logged_model.artifact_path
    model_uri = logged_model.model_uri

    # save_path
    save_path = root_path/"run_info.json"
    save_run_information(run_id,artifact_path,model_uri,save_path)
    logger.info('model info saved  successfully')

if __name__ == "__main__":
    main()