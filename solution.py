from fastapi import FastAPI
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
import os

app = FastAPI()

DATABASE_URL="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
FEATURES_DF_NAME="vladislav_lantsev_boodt_ftrs_ml_less_22"
CHUNKSIZE="200000"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
def load_features() -> pd.DataFrame:

    engine = create_engine(DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    try:

        print(("from sql - start loading"))
        for chunk_dataframe in pd.read_sql(FEATURES_DF_NAME,
                                           conn, chunksize=int(CHUNKSIZE)):

            chunks.append(chunk_dataframe)

        print(("from sql - loaded successfully"))

    except Exception as e:

        raise RuntimeError(f"Loading error: {e}")

    finally:
        conn.close()

    return pd.concat(chunks, ignore_index=True)
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

    # Obtain connection with target db
    user = pd.read_sql("SELECT * FROM public.user_data;", DATABASE_URL)
    print(user.head())
    return user

def get_post_df():

    # Obtain connection with target db
    post = pd.read_sql("SELECT * FROM public.post_text_df;", DATABASE_URL)
    print(post.head())
    return post

def get_user_features() -> pd.DataFrame:

    # Loading user table from the DB
    user = get_user_df()
    print(user.user_id.nunique())

    # Loading learning DF
    #data = pd.read_csv('df_to_learn.csv', sep=';')
    data = load_features()

    print(data.shape)
    print(data.head())
    print(data.columns)

    # Convert 'city' parameter to binary 'city_capital
    capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
                'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
    user['city'] = user.city.apply(lambda x: 1 if x in capitals else 0)
    user = user.rename(columns={"city": "city_capital"})

    # Remove unnecessary features
    user = user.drop(['os', 'source'], axis=1)

    # Convert all numbers to float32
    numeric_columns = user.select_dtypes(include=['float64', 'int64']).columns
    user[numeric_columns] = user[numeric_columns].astype('float32')

    # Add all users from the user table
    user = user.combine_first(data)

    # Numerical categorical to int32
    user['exp_group'] = user['exp_group'].astype('int32')

    print(user.shape)
    print(user.main_topic_liked.isna().sum())
    print(user.user_id.nunique())
    print(user.post_id.nunique())
    print(data.user_id.nunique())

    return user


# Set of the model features with correct order
columns = ['topic', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4',
            'cluster_5','cluster_6', 'cluster_7', 'cluster_8', 'cluster_9',
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

    # Taking user data by ID from the original user table
    user_features = user_df[user_df['user_id'] == id][['user_id',
                                                       'gender',
                                                       'age',
                                                       'country',
                                                       'exp_group',
                                                       'city_capital',
                                                       'main_topic_liked',
                                                       'main_topic_viewed',
                                                       'views_per_user',
                                                       'likes_per_user']]

    calc_features = ['main_topic_liked', 'main_topic_viewed', 'views_per_user', 'likes_per_user']

    for i in calc_features:

         # Filter empty records after merge with mode value (very few users)
        user_features[i] = user_features[i].apply(lambda x: user_features[i].mode()[0])

    user_features = user_features.iloc[0].to_frame().T.reset_index()

    # Набираю признаки из time
    user_features['hour'] = time.hour
    user_features['month'] = time.month
    user_features['day'] = time.day

    # Time indicator in hours from the beginning of 2021
    user_features['time_indicator'] = (time.year - 2021) * 360 * 24 + time.month * 30 * 24 + time.day * 24 + time.hour

    #logger.info(user_features)

    # Post pool for prediction: min likes and views as a filter
    post_pull = user_df[(user_df['user_id'] != user_features.iloc[0]['user_id'])
                     & (user_df['post_views'] > 20)
                     & (user_df['post_likes'] > 10)
                     ][['post_id',
                        'post_likes',
                        'post_views',
                        'text_length',
                        'cluster_1',
                        'cluster_2',
                        'cluster_3',
                        'cluster_4',
                        'cluster_5',
                        'cluster_6',
                        'cluster_7',
                        'cluster_8',
                        'cluster_9',
                        'topic'
                        ]].drop_duplicates('post_id').reset_index()


    # Merge with user data and preparing for prediction
    X = post_pull.combine_first(user_features).drop(['user_id', 'post_id'], axis=1)

    columns_to_fill = ['age',
                       'city_capital',
                       'country',
                       'day',
                       'exp_group',
                       'gender',
                       'hour',
                       'month',
                       'likes_per_user',
                       'main_topic_liked',
                       'main_topic_viewed',
                       'time_indicator',
                       'views_per_user'
                       ]
    for i in columns_to_fill:

        # filling gaps after combine_first with input user data
        X[i] = X[i].fillna(X[i].mode()[0])

    # All integers - to int32
    X[['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4',
       'cluster_5', 'cluster_6', 'cluster_7', 'cluster_8',
       'cluster_9', 'exp_group', 'month']] = X[
        ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4',
         'cluster_5','cluster_6', 'cluster_7', 'cluster_8',
         'cluster_9','exp_group', 'month']].astype('int32')

    # Features order - as at learning
    X = X[columns]

    # Like probability prediction
    X['ax'] = model.predict_proba(X)[:, 1]

    X = X.combine_first(post_pull)

    # First n=limit post with the highest like probability
    posts_recnd = X.drop_duplicates('post_id').sort_values(ascending=False, by='ax').head(limit)['post_id'].to_list()

    #logger.info(posts_recnd)

    posts_recnd_list = []

    # Getting post data by obtained IDs
    for i in posts_recnd:

        posts_recnd_list.append(PostGet(id=i,
                                        text=post_df[post_df['post_id'] == i].text.iloc[0],
                                        topic=post_df[post_df['post_id'] == i].topic.iloc[0])
                                )

    return posts_recnd_list
