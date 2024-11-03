from learn_model import learn_model
from get_features_table import df_to_sql, get_user_features, load_features
from get_predict_by_model import  check_model_locally
from dotenv import load_dotenv

#загрузка переменных окружения
load_dotenv()

#model, df, cat_columns = learn_model(1000000)

#df = get_user_features()
#df_to_sql(get_user_features())

check_model_locally()

#df = load_features()

#print(df.head(20))