from learn_model import learn_model
from get_features_table import df_to_sql, get_user_post_features, load_features
from dotenv import load_dotenv
import pandas as pd
import os

# Load env variables
load_dotenv()

# data, post, model = learn_model()

# user_df, post_df, nn_input_columns_df = get_user_post_features()
#
# df_to_sql(user_df, os.getenv('USER_FEATURES_DF_NAME'))
# df_to_sql(post_df, os.getenv('POST_FEATURES_DF_NAME'))
# df_to_sql(nn_input_columns_df, os.getenv('NN_INPUT_COLUMNS_DF_NAME'))


