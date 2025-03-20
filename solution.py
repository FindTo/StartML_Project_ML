from fastapi import FastAPI
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import os
import hashlib

app = FastAPI()

DATABASE_URL="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
CHUNKSIZE="200000"

USER_FEATURES_DF_NAME_NN="vladislav_lantsev_user_features_lesson_10_dl"
POST_FEATURES_DF_NAME_NN="vladislav_lantsev_post_features_lesson_10_dl"
NN_INPUT_COLUMNS_DF_NAME="vladislav_lantsev_nn_input_columns_lesson_10_dl"
NN_MODEL_NAME = "nn_estinmate_likes_200xPCA_embedds_1024k_825_drop_03_02.pt"



engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def load_features(features_name) -> pd.DataFrame:

    engine = create_engine(DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    try:

        print(("from sql - start loading"))
        for chunk_dataframe in pd.read_sql(features_name,
                                           conn, chunksize=int(CHUNKSIZE)):

            chunks.append(chunk_dataframe)

        print(("from sql - loaded successfully"))

    except Exception as e:

        raise RuntimeError(f"Loading error: {e}")

    finally:
        conn.close()

    return pd.concat(chunks, ignore_index=True)

# Class for NN-based recommendation
class NN_Recommend():

    class FirstModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(

                nn.Linear(245, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.3),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.3),

                nn.Linear(64, 16),
                nn.BatchNorm1d(16),
                nn.Sigmoid(),
                nn.Dropout(p=0.2),

                nn.Linear(16, 1),

            )

        def forward(self, x):
            return self.net(x)

    user_df: pd.DataFrame
    post_df: pd.DataFrame
    nn_input_columns_df: pd.DataFrame
    post_original_df: pd.DataFrame
    model: FirstModel

    def __init__(self, ):

        # Download DFs with user, post and input NNs columns from the DB before service activation
        self.user_df = load_features(USER_FEATURES_DF_NAME_NN)
        self.post_df = load_features(POST_FEATURES_DF_NAME_NN)
        self.nn_input_columns_df = load_features(NN_INPUT_COLUMNS_DF_NAME)
        self.post_original_df = get_post_df()
        # Load NNs weights
        self.model = load_models(1)

    def apply(self, id: int, time: datetime, limit: int = 5):

        # Finding user in the prepared df
        user_features =self.user_df[self.user_df['user_id'] == id].reset_index(drop=True)

        # Getting time-related features
        user_features['hour'] = time.hour
        user_features['day'] = time.day

        # Only 3 months (10, 11, 12) available

        if time.month == 11:

            user_features['month_11'] = 1.0
            user_features['month_12'] = 0.0

        elif time.month == 12:

            user_features['month_11'] = 0.0
            user_features['month_12'] = 1.0

        else:

            user_features['month_11'] = 0.0
            user_features['month_12'] = 0.0

        # Time indicator from the beginning of 2021 up to now
        user_features['time_indicator'] = (
                                                      time.year - 2021) * 360 * 24 + time.month * 30 * 24 + time.day * 24 + time.hour

        # logger.info(user_features)

        # Post pull filtered by likes and views
        post_pull = self.post_df[(self.post_df['post_views'] > 100) & (self.post_df['post_likes'] > 10)]

        # Merge post pull with user vector and fill the gaps
        X = post_pull.combine_first(user_features)

        # logger.info(X.head())

        # Fill the gaps with user features from the first row
        for col in user_features.columns.to_list():
            X[col] = X[col].iloc[0]

        # Drop unnecessary IDs, indexes are by post df
        X.drop(['post_id', 'user_id'], axis=1, inplace=True)

        # Arrange column in accordance with NN input
        X = X[self.nn_input_columns_df['0'].to_list()]

        # Convert df to tensor and make predictions using NN model
        X_tens = torch.FloatTensor(X.values)
        X['ax'] = torch.sigmoid(self.model(X_tens)).detach().numpy().astype("float32")

        # Return post_id columns
        X = X.combine_first(post_pull)

        # First n=limit posts from pull with max like probability
        posts_recnd = X.sort_values(ascending=False, by='ax').head(limit)['post_id'].to_list()

        # logger.info(posts_recnd)

        posts_recnd_list = []

        # Making response by Pydantic using the obtained post IDs
        for i in posts_recnd:
            posts_recnd_list.append(PostGet(id=i,
                                            text=self.post_original_df[self.post_df['post_id'] == i].text.iloc[0],
                                            topic=self.post_original_df[self.post_df['post_id'] == i].topic.iloc[0])
                                    )

        return posts_recnd_list


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path

    return MODEL_PATH

def load_models(group:int):

    if group == 1:

        model_path = get_model_path(NN_MODEL_NAME)
        model = NN_Recommend.FirstModel()

        model.load_state_dict(torch.load(model_path,
                        map_location=torch.device('cpu')))
        model.eval()

    if group == 0:



    return model


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

# A/B testing users separation: 0 is the control group (A), 1 is the test group (B)
def get_exp_group(user_id: int) -> int:

    salt = 'a_b_testing'
    user = str(user_id)
    value_str = user + salt
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    return value_num % 2

def get_user_df():

    # Obtain db connection
    user = pd.read_sql("SELECT * FROM public.user_data;", DATABASE_URL)
    print(user.head())
    return user

def get_post_df():

    # Obtain db connection
    post = pd.read_sql("SELECT * FROM public.post_text_df;", DATABASE_URL)
    print(post.head())
    return post

def get_db():
    with SessionLocal() as db:
        return db

# Creating NN recommend model instance - downloading all the necessary files from the DB
nn_recommend = NN_Recommend()

# Creating gradient boosting recommend model instance - downloading all the necessary files from the DB
boost_recommend = Boost_Recommend()

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:

    exp_group = get_exp_group(id)

    if exp_group == 0:
        posts_recnd_list = nn_recommend.apply(id, time, limit)

    elif exp_group == 1:
        posts_recnd_list = boost_recommend.apply(id, time, limit)

    else:
        raise ValueError('unknown group')

    return Response(str(exp_group), posts_recnd_list)
