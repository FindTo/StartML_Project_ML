from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from database import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import desc
from table_post import Post
from table_user import User
from table_feed import Feed
from schema import UserGet, PostGet, FeedGet
from typing import List
from datetime import datetime
from get_features_table import get_user_features
from learn_model import get_post_df
from get_predict_by_model import load_models


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

@app.get("/user/{id}", response_model = UserGet)
def get_user(id: int, db: Session = Depends(get_db)):

    data = db.query(User).filter(User.id == id).first()

    if data == None:

        raise HTTPException(404, "user not found")

    else:
        logger.info(data)
        return data

@app.get("/post/{id}", response_model = PostGet)
def get_post(id: int, db: Session = Depends(get_db)):

    data = db.query(Post).filter(Post.id == id).first()

    if data == None:

        raise HTTPException(404, "post not found")

    else:

        return data

@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):

    data = db.query(Feed).filter(Feed.user_id == id).order_by(desc(Feed.time)).limit(limit).all()
    logger.info(data)

    return data

@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):

    return db.query(Feed).filter(Feed.post_id == id).order_by(desc(Feed.time)).limit(limit).all()

# @app.get("/post/recommendations/", response_model=List[PostGet])
# def get_post_recomended(id: int, limit: int = 10, db: Session = Depends(get_db)):
#
#     post_liked = db.query(Post, func.count(Feed.user_id)).select_from(Feed
#                           ).join(Post, Feed.post_id == Post.id).filter(Feed.action == 'like'
#                                    ).group_by(Post
#                                               ).order_by(desc(func.count(Feed.user_id))).limit(limit).all()
#
#     return [x[0] for x in post_liked]

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
