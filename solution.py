from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from database import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
import os

DATABASE_URL="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
FEATURES_DF_NAME="vladislav_lantsev_features_lesson_22"
CHUNKSIZE="200000"

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

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True
def get_user_df():

    # Установка соединения с базой данных
    user = pd.read_sql("SELECT * FROM public.user_data;", DATABASE_URL)
    print(user.head())
    return user

def get_post_df():

    # Установка соединения с базой данных
    post = pd.read_sql("SELECT * FROM public.post_text_df;", DATABASE_URL)
    print(post.head())
    return post

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

    # user['main_topic_liked'].fillna(user['main_topic_liked'].mode(), inplace=True)
    # user['main_topic_viewed'].fillna(user['main_topic_viewed'].mode(), inplace=True)
    # user['views_per_user'].fillna(user['views_per_user'].median(), inplace=True)
    # user['likes_per_user'].fillna(user['likes_per_user'].median(), inplace=True)

    print(user.shape)
    print(user.main_topic_liked.isna().sum())
    print(user.user_id.nunique())
    print(user.post_id.nunique())
    print(data.user_id.nunique())
    #user.sample(100).to_csv('user_data.csv', sep=';', index=False)
    #user.to_csv('vladislav_lantsev_features_lesson_22.csv', sep=';', index=False)

    return user
app = FastAPI()

# Список признаков модели
columns = ['topic', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4',
           'text_length', 'gender', 'age', 'country', 'exp_group', 'city_capital',
           'post_likes', 'post_views', 'hour', 'month', 'day', 'time_indicator',
           'main_topic_liked', 'main_topic_viewed', 'views_per_user',
           'likes_per_user']

user_df = get_user_features()

post_df = get_post_df()

model = load_models()


def get_db():
    with SessionLocal() as db:
        return db

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:

    # Выбираю записи из таблицы под заданного юзера
    user_features = user_df[user_df['user_id'] == id][['user_id', 'gender', 'age', 'country',
                                                   'exp_group', 'city_capital', 'main_topic_liked', 'main_topic_viewed',
                                                   'views_per_user',
                                                   'likes_per_user']]

    # Фильтр на грязные даные после мержа df и user
    user_features['main_topic_liked'] = user_features['main_topic_liked'].apply(
        lambda x: user_features['main_topic_liked'].mode()[0])
    user_features['main_topic_viewed'] = user_features['main_topic_viewed'].apply(
        lambda x: user_features['main_topic_viewed'].mode()[0])
    user_features['views_per_user'] = user_features['views_per_user'].apply(
        lambda x: user_features['views_per_user'].mode()[0])
    user_features['likes_per_user'] = user_features['likes_per_user'].apply(
        lambda x: user_features['likes_per_user'].mode()[0])

    user_features = user_features.iloc[0].to_frame().T.reset_index()

    # Набираю признаки из time
    user_features['hour'] = time.hour
    user_features['month'] = time.month
    user_features['day'] = time.day

    # Фича-индикатор суммарного времени с 2021 года до текущего момента просмотра, в часах
    user_features['time_indicator'] = (time.year - 2021) * 360 * 24 + time.month * 30 * 24 + time.day * 24 + time.hour

    logger.info(user_features)

    # Набираю пул постов для предсказания: для юзера они должны быть незнакомы и более-менее пролайканы и просмотрены
    post_pull = user_df[(user_df['user_id'] != user_features.iloc[0]['user_id'])
                     & (user_df['post_views'] > 60)
                     & (user_df['post_likes'] > 20)
                     ][['post_id', 'post_likes', 'post_views', 'text_length',
                        'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'topic'
                        ]].drop_duplicates('post_id').reset_index()

    # Мержу запись по юзеру с постами и заполняю пропуски данными юзера
    X = post_pull.combine_first(user_features).drop(['user_id', 'post_id'], axis=1)
    X['age'] = X['age'].fillna(X['age'].mode()[0])
    X['city_capital'] = X['city_capital'].fillna(X['city_capital'].mode()[0])
    X['country'] = X['country'].fillna(X['country'].mode()[0])
    X['city_capital'] = X['city_capital'].fillna(X['city_capital'].mode()[0])
    X['day'] = X['day'].fillna(X['day'].mode()[0])
    X['exp_group'] = X['exp_group'].fillna(X['exp_group'].mode()[0])
    X['gender'] = X['gender'].fillna(X['gender'].mode()[0])
    X['hour'] = X['hour'].fillna(X['hour'].mode()[0])
    X['month'] = X['month'].fillna(X['month'].mode()[0])
    X['likes_per_user'] = X['likes_per_user'].fillna(X['likes_per_user'].mode()[0])
    X['main_topic_liked'] = X['main_topic_liked'].fillna(X['main_topic_liked'].mode()[0])
    X['main_topic_viewed'] = X['main_topic_viewed'].fillna(X['main_topic_viewed'].mode()[0])
    X['time_indicator'] = X['time_indicator'].fillna(X['time_indicator'].mode()[0])
    X['views_per_user'] = X['views_per_user'].fillna(X['views_per_user'].mode()[0])

    # Привожу к формату признаков модели
    X[['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'exp_group', 'month']] = X[
        ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'exp_group', 'month']].astype('int32')

    X = X[columns]

    X['ax'] = model.predict_proba(X)[:, 1]

    X = X.combine_first(post_pull)

    # Первые n=limit постов из пула с максимальной вероятностью лайка
    posts_recnd = X.drop_duplicates('post_id').sort_values(ascending=False, by='ax').head(limit)['post_id'].to_list()

    logger.info(posts_recnd)

    posts_recnd_list = []

    # Набираю посты из скачанной таблицы постов
    for i in posts_recnd:

        posts_recnd_list.append(PostGet(id=i,
                                        text=post_df[post_df['post_id'] == i].text.iloc[0],
                                        topic=post_df[post_df['post_id'] == i].topic.iloc[0])
                                )

    return posts_recnd_list
