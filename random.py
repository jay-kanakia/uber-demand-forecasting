# initialize dagashub
dagshub.init(repo_owner='jay-kanakia', repo_name='uber-demand-forecasting', mlflow=True)

# set mlflow tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/uber-demand-forecasting.mlflow')

# set mlflow experiment name
mlflow.set_experiment('DVC Pipeline')