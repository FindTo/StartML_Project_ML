import pandas as pd
from catboost import CatBoostClassifier
from learn_model import calculate_hitrate
import os

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model_final_proj")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def check_model_locally(n: int = 1000):

    from_file = CatBoostClassifier()
    from_file.load_model("catboost_model_final_proj")

    data = pd.read_csv('df_to_learn.csv', sep=';')

    df = data.sample(n)
    X = df.drop(['user_id', 'target', 'post_id'], axis=1)
    y = df.target

    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_columns] = X[numeric_columns].astype('float32')

    X[['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4',
       'cluster_5','cluster_6', 'cluster_7', 'cluster_8', 'cluster_9',
       'exp_group', 'month']] = X[
        ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4',
         'cluster_5','cluster_6', 'cluster_7', 'cluster_8', 'cluster_9',
         'exp_group', 'month']].astype('int32')

    print(from_file.predict_proba(X)[:, 1])

    hitrate = calculate_hitrate(y.values, from_file.predict_proba(X)[:, 1], k=5)
    print(f'Hitrate для бустинга на тесте: {hitrate}')

    return 0

