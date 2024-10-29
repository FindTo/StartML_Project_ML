import pandas as pd
from learn_model import get_user_df, get_post_df

user = get_user_df()

data = pd.read_csv('df_to_learn.csv', sep=';')
data = data.drop_duplicates()
print(data.shape)
print(data.head())
print(data.columns)

# Добавил булевый признак по главным городам в представленных странах, остальных городов слишком много
capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
            'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
user['city'] = user.city.apply(lambda x: 1 if x in capitals else 0)
user = user.rename(columns={"city": "city_capital"})

# Выбираем только числовые столбцы для преобразования
numeric_columns = user.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
user[numeric_columns] = user[numeric_columns].astype('float32')

# Преобразуем численные категориальные в int32
user[['exp_group']] = user[['exp_group']].astype('int32')


user_features = data[['user_id', 'gender', 'age', 'country',
       'exp_group', 'city_capital', 'main_topic_liked', 'main_topic_viewed', 'views_per_user',
       'likes_per_user']]

print(user_features.shape)

user = user.merge(user_features, on='user_id', how='left')
user.sample(100).to_csv('user_data.csv', sep=';', index=False)

print(user.sample(100))
print(user.columns)
print(user.shape)

