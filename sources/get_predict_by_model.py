import pandas as pd
from catboost import CatBoostClassifier
from learn_model import calculate_hitrate

from_file = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть

from_file.load_model("catboost_model_final_proj")

data = pd.read_csv('df_to_learn.csv', sep=';')

df = data.sample(100)
X = df.drop(['user_id', 'target'], axis=1)
y = df.target

# Выбираем только числовые столбцы для преобразования
numeric_columns = X.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
X[numeric_columns] = X[numeric_columns].astype('float32')

# Преобразуем численные категориальные в int32
X[['exp_group', 'month']] = X[['exp_group', 'month']].astype('int32')

print(from_file.predict_proba(X)[:, 1])

hitrate = calculate_hitrate(y.values, from_file.predict_proba(X)[:, 1], k=5)
print(f'Hitrate для бустинга на тесте: {hitrate}')
