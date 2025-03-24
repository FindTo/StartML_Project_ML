import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, KMeans
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt
import os

def get_user_df():

    user = pd.read_sql("SELECT * FROM public.user_data;", os.getenv('DATABASE_URL'))
    print(user.head())
    return user


def get_post_df():

    post = pd.read_sql("SELECT * FROM public.post_text_df;", os.getenv('DATABASE_URL'))
    print(post.head())
    return post


# Preparing features using DB tables for boosting model
def get_features_df(feed_n_lines=1000000):

    # Downloading tables from the DB
    user = get_user_df()
    post = get_post_df()
    feed = pd.read_sql(f"SELECT * FROM public.feed_data order by random() LIMIT {feed_n_lines};", os.getenv('DATABASE_URL'))
    feed = feed.drop_duplicates()
    print(feed.head())

    # Processing categorical features
    new_user = user.drop('city', axis=1)

    categorical_columns = []
    categorical_columns.append('country')
    #categorical_columns.append('os')
    #categorical_columns.append('source')
    categorical_columns.append('exp_group')

    #print(categorical_columns)

    # for col in categorical_columns:
    #     one_hot = pd.get_dummies(new_user[col], prefix=col, drop_first=True, dtype='int64')
    #
    #     new_user = pd.concat((new_user.drop(col, axis=1), one_hot), axis=1)

    # Boolean feature 'city_capital' indicates is the city main in the country

    capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
                'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
    cap_bool = user.city.apply(lambda x: 1 if x in capitals else 0)
    new_user = pd.concat([new_user, cap_bool], axis=1, join='inner')
    new_user = new_user.rename(columns={"city": "city_capital"})

    # Choosing numerical features
    numeric_columns = new_user.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns

    # Convert numerical into float32
    new_user[numeric_columns] = new_user[numeric_columns].astype('float32')

    # Original user features are ready
    new_user.head()
    num_user_full = new_user['user_id'].nunique()
    print(f'Число уникальных юзеров:{num_user_full}')

    num_post_full = post['post_id'].nunique()
    print(f'Число уникальных постов:{num_post_full}')

    # TF-IDF feature creation: getting top 2000 words by absolute metric value
    tfidf = TfidfVectorizer(stop_words='english', strip_accents='unicode', min_df = 0.01, max_features = 2000)
    tfidf_matrix = tfidf.fit_transform(post['text'].fillna('unknown'))
    #feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.reset_index(drop=True, inplace=True)
    post.reset_index(drop=True, inplace=True)

    # Выделяем главные значаящие слова по методу PCA - от максимальной дисперсии
    # Далее берем топ значимых TF-IDF для PCA фичей, и по ним делаем кластеризацию
    # Это будет новый категориальный признак

    # PCA based on TF-IDF: making components based on TF-IDF dataframe for 2000 words
    # Centering data using standard scaler for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(tfidf_df)

    # Apply PCA
    pca = PCA(n_components=200)
    X_pca = pca.fit_transform(X_scaled)
    X_pca = pd.DataFrame(X_pca)
    X_pca = X_pca.add_prefix('PCA_')

    # Coefficients by original PCA components
    components = pca.components_

    # Choosing top features from TF-IDF by importance
    # Create importance dataframe
    importance_df = pd.DataFrame(tfidf_df, columns=tfidf.get_feature_names_out())

    # Summ TF-IDF absolute metrics by rows
    importance_scores = importance_df.abs().sum(axis=0)

    # Sort features (words) by importance using summarized metrics
    sorted_importance = importance_scores.sort_values(ascending=False)

    # Sorted top of features
    top_features = sorted_importance[:1000].index.tolist()

    print("Most important words:", top_features)

    # Using clusterization: making new features from TF-IDF matrix by the most important words
    #clustering = DBSCAN(eps=0.5, min_samples=4).fit_predict(tfidf_df[top_features])
    clustering = KMeans(n_clusters=10, random_state=42, max_iter=1000).fit_predict(tfidf_df[top_features])
    clusters = pd.DataFrame({"Clusters": clustering})

    print(clusters.nunique())

    # categorical_columns.append('clusters')

    # OHE for clusters (manually) and append to 'post' df
    one_hot = pd.get_dummies(clusters['clusters'], prefix='cluster', drop_first=True, dtype='int32')

    post = pd.concat((post, one_hot), axis=1)

    # Additional 'post' related features
    # Length of post
    post['text_length'] = post['text'].apply(len)

    # Remove original texts from df
    post = post.drop(['text'], axis=1)

    # Mark 'topic' feature ac categorical
    categorical_columns.append('topic')

    # Choosing numerical features in 'post' df
    numeric_columns = post.select_dtypes(include=['float64', 'int64']).columns

    # Convert numerical to float32
    post[numeric_columns] = post[numeric_columns].astype('float32')

    # Choosing numerical features in 'feed' df
    numeric_columns = feed.select_dtypes(include=['float64', 'int64' ]).columns

    # Convert numerical to float32
    feed[numeric_columns] = feed[numeric_columns].astype('float32')

    # Rename 'action' to 'action_class'
    feed = feed.rename(columns={"action": "action_class"})

    # Merge 'post' with 'feed' and 'new_user'
    df = pd.merge(
        feed,
        post,
        on='post_id',
        how='left'
    )

    df = pd.merge(
        df,
        new_user,
        on='user_id',
        how='left'
    )
    df.head()

    # New feature: likes counter for posts using data from 'feed'
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)
    df['post_likes'] = df.groupby('post_id')['action_class'].transform('sum')

    # New feature: views counter for posts using data from 'feed'
    #df['views_per_post'] = df.groupby('post_id')['action_class'].apply(lambda x: 1 if x == 0 else 0).transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
    df['post_views'] = df.groupby('post_id')['action_class'].transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

    # Getting datetime from 'timestamp' feature
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort DF by datetime (historical format)
    df = df.sort_values('timestamp')

    # New features from 'timestamp'
    df['day_of_week'] = df.timestamp.dt.dayofweek
    df['hour'] = df.timestamp.dt.hour
    df['month'] = df.timestamp.dt.month
    df['day'] = df.timestamp.dt.day
    df['year'] = df.timestamp.dt.year

    # New feature - summarized time in hours from the beginning of 2021
    df['time_indicator'] = (df['year'] - 2021)*360*24 + df['month']*30*24 + df['day']*24 + df['hour']

    categorical_columns.append('month')
    #categorical_columns.append('year')

    # OneHotEncoding by 'month'
    # one_hot = pd.get_dummies(df['month'], prefix='month', drop_first=True, dtype='int32')
    #
    # df = pd.concat((df.drop('month', axis=1), one_hot), axis=1)

    # New user features: top topics for users by their history of views/likes in 'feed'
    main_liked_topics = df[df['action_class'] == 1].groupby(['user_id'])['topic'].agg(lambda x: np.random.choice(x.mode())).to_frame().reset_index()
    main_liked_topics = main_liked_topics.rename(columns={"topic": "main_topic_liked"})
    main_viewed_topics = df[df['action_class'] == 0].groupby(['user_id'])['topic'].agg(lambda x: np.random.choice(x.mode())).to_frame().reset_index()
    main_viewed_topics = main_viewed_topics.rename(columns={"topic": "main_topic_viewed"})

    # Merge new user features with master df
    df = pd.merge(df, main_liked_topics,  on='user_id', how='left')
    df = pd.merge(df, main_viewed_topics, on='user_id', how='left')

    # Fill empty spaces with the most common value
    df['main_topic_liked'].fillna(df['main_topic_liked'].mode().item(), inplace=True)
    df['main_topic_viewed'].fillna(df['main_topic_viewed'].mode().item(), inplace=True)

    # New topic-related features - mark as categorical
    categorical_columns.append('main_topic_viewed')
    categorical_columns.append('main_topic_liked')

    # New user feature: number of likes per user from 'feed'
    likes_per_user = df.groupby(['user_id'])['action_class'].agg(pd.Series.sum).to_frame().reset_index()
    likes_per_user = likes_per_user.rename(columns={"action_class": "likes_per_user"})

    # New user feature: number of likes per user from 'feed'
    #df['views_per_user'] = df.groupby('user_id')['action_class'].apply(lambda x: 1 if x == 0 else 0).transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
    df['views_per_user'] = df.groupby('user_id')['action_class'].transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

    # Concat new user feature with master df
    df = pd.merge(df, likes_per_user,  on='user_id',how='left')

    # Choosing numerical features and convert it in float32
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].astype('float32')

    # Looking at the numbers of engaged users/posts
    num_user_df = df['user_id'].nunique()
    print(f'Unique users count: {num_user_df}')
    num_post_df = df['post_id'].nunique()
    print(f'Unique posts count::{num_post_df}')

    # Set target as 1 if action class means like
    df['target'] = df['target'].astype('int32')
    df['action_class'] = df['action_class'].astype('int32')
    df['target'] = df['target'] | df['action_class']

    # Remove unnecessary features
    df = df.drop(['timestamp', 'action_class', 'os', 'source', 'day_of_week', 'year'], axis=1)

    # Convert numerical categorical to int32
    df[['exp_group', 'month']] = df[['exp_group', 'month']].astype('int32')

    # Save master DF to csv with all IDs
    df.to_csv('df_to_learn.csv', sep=';', index=False)

    # Remove IDs
    df = df.drop(['user_id', 'post_id'], axis=1)

    # Calculate master DF size
    total_memory = df.memory_usage(deep=True).sum()
    print(f"\nMemory size for the master DataFrame: {total_memory} byte")
    print(df.dtypes)

    return df, categorical_columns

# Local hitrate by master DF
def calculate_hitrate(y_true, y_pred_proba, k=5):
    hits = 0
    n = len(y_true)

    for i in range(n):
        # Top-k probabilities indices by user
        top_k_indices = np.argsort(y_pred_proba[i])[-k:]

        # Check if one of predictions is in liked
        if any(y_true[i] == 1 for idx in top_k_indices):
            hits += 1

    # Return the total hit rate
    hitrate = hits / n

    return hitrate
def learn_model(df_size = 1000000):

    # Getting DF for leaning
    data, cat_columns = get_features_df(feed_n_lines=df_size)

    X = data.drop('target', axis=1)
    y = data.target

    # Separate train and test samples without shuffle (historical hierarchy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    search = CatBoostClassifier(verbose=False,
                                depth=6,
                                learning_rate=0.2,
                                iterations=200,
                                l2_leaf_reg=2000,
                                cat_features=cat_columns)

    search.fit(X_train, y_train)

    # Save obtained model foe boosting
    search.save_model('catboost_model_final_proj', format="cbm")

    # Feature importance graph
    feature_imp = search.feature_importances_

    forest_importances = pd.Series(feature_imp, index=X.columns)
    fig, ax = plt.subplots()
    forest_importances.plot.bar()
    ax.set_title("Feature importances Catboost")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # F-measures train/test
    f1_loc_tr = round(f1_score(y_train, search.predict(X_train), average='weighted'), 5)
    f1_loc_test = round(f1_score(y_test, search.predict(X_test), average='weighted'), 5)
    print(f'F-measure for boosting at train: {f1_loc_tr}')
    print(f'F-measure for boosting at test: {f1_loc_test}')

    # AUC train
    fpr, tpr, thd = roc_curve(y_train, search.predict_proba(X_train)[:, 1])
    print(f'AUC for boosting at train: {auc(fpr, tpr):.5f}')

    # AUC test
    fpr, tpr, thd = roc_curve(y_test, search.predict_proba(X_test)[:, 1])
    print(f'AUC for boosting at test: {auc(fpr, tpr):.5f}')

    # Local hitrate by learning DF
    hitrate = calculate_hitrate(y_test.values, search.predict_proba(X_test)[:, 1], k = 5)

    print(f'Local hitrate for boosting at test: {hitrate}')

    # ROC curve graph for test
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()

    return search, data, cat_columns
