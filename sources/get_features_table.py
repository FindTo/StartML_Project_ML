import pandas as pd
from learn_model import get_user_df
from sqlalchemy import create_engine
import os

# Get full DF with all users, relying on the DF for model learning
def get_user_features() -> pd.DataFrame:

    # Download original 'user' dataframe
    user = get_user_df()
    print(user.user_id.nunique())

    # Download master DF for learning
    #data = pd.read_csv('df_to_learn.csv', sep=';')
    data = load_features()

    print(data.shape)
    print(data.head())
    print(data.columns)

    # Create 'city' boolean feature as for the learning DF
    capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
                'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
    user['city'] = user.city.apply(lambda x: 1 if x in capitals else 0)
    user = user.rename(columns={"city": "city_capital"})

    # Remove unnecessary features
    user = user.drop(['os', 'source'], axis=1)

    # Convert numerical features to float32
    numeric_columns = user.select_dtypes(include=['float64', 'int64']).columns
    user[numeric_columns] = user[numeric_columns].astype('float32')

    # Merge 'user' df and master df from learning
    user = user.combine_first(data)

    # Convert numerical categorical to int32
    user['exp_group'] = user['exp_group'].astype('int32')

    print(user.shape)
    print(user.main_topic_liked.isna().sum())
    print(user.user_id.nunique())
    print(user.post_id.nunique())
    print(data.user_id.nunique())

    return user

def df_to_sql(df):

    # Try to write DF as a single instance
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

# Load big DF from the DB using chunks
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

        raise RuntimeError(f"Data loading error: {e}")

    finally:
        conn.close()

    return pd.concat(chunks, ignore_index=True)

