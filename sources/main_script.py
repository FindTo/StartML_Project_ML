from learn_model import learn_model
from get_features_table import df_to_sql, get_user_features, load_features
from get_predict_by_model import  check_model_locally
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

#model, df, cat_columns = learn_model(1024000)

#df = pd.read_csv("df_to_learn.csv", sep=';')

#df_to_sql(df)

#check_model_locally()

