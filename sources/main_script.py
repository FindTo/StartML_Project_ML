from learn_model import learn_model
from get_features_table import df_to_sql, get_user_features, load_features
from dotenv import load_dotenv

#загрузка переменных окружения
load_dotenv()

#model, df, cat_columns = learn_model(2500000)

df_to_sql(get_user_features())

df = load_features()

print(df.head())