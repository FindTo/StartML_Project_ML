import pandas as pd
from learn_model import get_user_df
from sqlalchemy import create_engine
import os

# Получить полную таблицу с фичами на всех юзеров, опираясь на датафрейм с обучения
def get_user_features() -> pd.DataFrame:
    # Выгружаем таблицу user с сервера
    user = get_user_df()
    print(user.user_id.nunique())

    # Читаем датафрейм с обучения модели
    data = pd.read_csv('df_to_learn.csv', sep=';')

    print(data.shape)
    print(data.head())
    print(data.columns)

    # Так же в скачанном user обновляю параметр city
    capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
                'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
    user['city'] = user.city.apply(lambda x: 1 if x in capitals else 0)
    user = user.rename(columns={"city": "city_capital"})

    # Убераю лишние для модели признаки
    user = user.drop(['os', 'source'], axis=1)

    # Конвертация численных во float32 для модели
    numeric_columns = user.select_dtypes(include=['float64', 'int64']).columns
    user[numeric_columns] = user[numeric_columns].astype('float32')

    # user_features = data[['user_id', 'gender', 'age', 'country',
    #        'exp_group', 'city_capital', 'main_topic_liked', 'main_topic_viewed', 'views_per_user',
    #        'likes_per_user']]

    # Объединение таблицы с обучения вместе с таблицой user, чтобы иметь всех юзеров
    user = user.combine_first(data)

    # Конвертация категориального численного в int32 для модели
    user['exp_group'] = user['exp_group'].astype('int32')

    # user = user.drop(['gender_x', 'age_x', 'country_x', 'city_capital_x','exp_group_x',],
    #                  axis = 1).rename(columns={'gender_y': 'gender', 'age_y': 'age', 'country_y': 'country',
    #                                         'exp_group_y': 'exp_group', 'city_capital_y': 'city_capital'})

    # # Преобразуем численные категориальные в int32
    # user[['exp_group']] = user[['exp_group']].astype('int32')

    # print(user[['gender_y', 'age_y', 'country_y', 'city_capital_y', 'exp_group_y']].combine_first(user[['gender_x',
    #                                                                                             'age_x',
    #                                                                                             'country_x',
    #                                                                                             'city_capital_x',
    #                                                                                             'exp_group_x']]).shape)
    # user['main_topic_liked'].fillna(user['main_topic_liked'].mode(), inplace=True)
    # user['main_topic_viewed'].fillna(user['main_topic_viewed'].mode(), inplace=True)
    # user['views_per_user'].fillna(user['views_per_user'].median(), inplace=True)
    # user['likes_per_user'].fillna(user['likes_per_user'].median(), inplace=True)

    print(user.shape)
    #print(user.columns)
    print(user.main_topic_liked.isna().sum())
    print(user.user_id.nunique())
    print(user.post_id.nunique())
    print(data.user_id.nunique())
    user.sample(100).to_csv('user_data.csv', sep=';', index=False)
    user.to_csv('vladislav_lantsev_features_lesson_22.csv', sep=';', index=False)

    return user

def df_to_sql(df):

    # Пробую записать таблицу в Sql
    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)

    try:

        print(("to_sql - start writing"))
        df.to_sql(os.getenv('FEATURES_DF_NAME'), con=engine, if_exists='replace', index=False)
        print(("to_sql - successfully written"))
        conn.close()

    except:

        print(("to_sql - failed to write"))
        conn.close()

    return 0

def load_features() -> pd.DataFrame:

    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    try:

        print(("from sql - start loading"))
        for chunk_dataframe in pd.read_sql(os.getenv('FEATURES_DF_NAME'),
                                           conn, chunksize=int(os.getenv('CHUNKSIZE'))):

            chunks.append(chunk_dataframe)

        print(("from sql - loaded successfully"))

    except Exception as e:

        raise RuntimeError(f"Ошибка при загрузке данных: {e}")

    finally:
        conn.close()

    return pd.concat(chunks, ignore_index=True)

