import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
from sklearn.utils import resample
import os

# Установка соединения с базой данных
conn_uri = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
user_data = pd.read_sql("SELECT * FROM public.user_data;", conn_uri)
post_text = pd.read_sql("SELECT * FROM public.post_text_df;", conn_uri)
feed_data = pd.read_sql("SELECT * FROM public.feed_data LIMIT 1000000;", conn_uri)
tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
tfidf_matrix = tfidf.fit_transform(post_text['text'].fillna('unknown'))
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
tfidf_df.reset_index(drop=True, inplace=True)
post_text.reset_index(drop=True, inplace=True)
tfidf_df.head()

from sklearn.decomposition import PCA

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(tfidf_matrix.toarray())

# Применение PCA
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

# Коэффициенты главных компонент
components = pca.components_

# Создание DataFrame для значимости
importance_df = pd.DataFrame(components, columns=tfidf.get_feature_names_out())

# Суммирование абсолютных значений коэффициентов для каждой колонки
importance_scores = importance_df.abs().sum(axis=0)

# Сортировка по значимости
sorted_importance = importance_scores.sort_values(ascending=False)

# Получение имен колонок, отсортированных по значимости
top_features = sorted_importance.index.tolist()

print("100 наиболее значимых колонок:", top_features[:100])

post_text = pd.concat([post_text, tfidf_df[top_features[:100]]], axis=1)

# Обработка данных
feed_data['timestamp'] = pd.to_datetime(feed_data['timestamp'])
feed_data['day_of_week'] = feed_data.timestamp.dt.dayofweek
feed_data['hour'] = feed_data.timestamp.dt.hour
feed_data['month'] = feed_data.timestamp.dt.month
feed_data['day'] = feed_data.timestamp.dt.day

# Средняя длина текста поста
post_text['text_length'] = post_text['text'].apply(len)
# Популярность поста
feed_data['post_likes'] = feed_data.groupby('post_id')['target'].transform('sum')
# Создание признаков на основе взаимодействий с топиками
topic_interactions = feed_data.merge(post_text[['post_id', 'topic']], left_on='post_id', right_on='post_id', how='left')
topic_interactions = topic_interactions.groupby(['user_id', 'topic']).size().unstack(fill_value=0)

feed_data = feed_data.drop('timestamp', axis=1)
post_text = post_text.drop('text', axis=1)

# Выбираем только числовые столбцы для преобразования
numeric_columns = feed_data.select_dtypes(include=['float64', 'int64']).columns

# Преобразуем только числовые столбцы в float32
feed_data[numeric_columns] = feed_data[numeric_columns].astype('float32')

# Выбираем только числовые столбцы для преобразования
numeric_columns = post_text.select_dtypes(include=['float64', 'int64']).columns

# Преобразуем только числовые столбцы в float32
post_text[numeric_columns] = post_text[numeric_columns].astype('float32')

# Выбираем только числовые столбцы для преобразования
numeric_columns = user_data.select_dtypes(include=['float64', 'int64']).columns

# Преобразуем только числовые столбцы в float32
user_data[numeric_columns] = user_data[numeric_columns].astype('float32')


df = pd.merge(user_data, feed_data, on='user_id', how='left')
df = pd.merge(df, post_text, on='post_id', how='left')
df = pd.merge(df, topic_interactions, on='user_id', how='left')

# Кодирование категориальных переменных df
for col in df.select_dtypes(include=['object']).columns:
    if col != 'text':
        if df[col].nunique() < 5:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat((df.drop(col, axis=1), one_hot), axis=1)
        else:
            mean_target = df.groupby(col)['target'].mean()
            df[col] = df[col].map(mean_target)

# Выбираем только числовые столбцы для преобразования
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Преобразуем только числовые столбцы в float32
df[numeric_columns] = df[numeric_columns].astype('float32')


df.head()


# Получение общего объема памяти, занимаемой DataFrame
total_memory = df.memory_usage(deep=True).sum()
print(f"\nОбщий объем памяти, занимаемой DataFrame: {total_memory} байт")



df_catboost = df.dropna()

# Подготовка данных для обучения
X = df_catboost.drop('target', axis=1)
y = df_catboost['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки с учетом стратификации
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



def calculate_hitrate(y_true, y_pred_proba, k=5):
    hits = 0
    n = len(y_true)

    for i in range(n):
        # Получаем индексы топ-k вероятностей для текущего пользователя
        top_k_indices = np.argsort(y_pred_proba[i])[-k:]  # топ-k для текущего пользователя

        # Проверяем, попадает ли хотя бы одно предсказанное значение в класс 1 (лайк)
        if any(y_true[i] == 1 for idx in top_k_indices):  # Сравниваем класс y_true с топ-k предсказаний
            hits += 1

    # Возвращаем долю пользователей, для которых был хотя бы один лайк в топ-k
    hitrate = hits / n
    return hitrate


from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV


# Обучение модели CatBoost с RandomSearch
def train_catboost(X_train, y_train):
    catboost_model = CatBoostClassifier(verbose=0)
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.1, 0.2, 0.3],
        'iterations': [100, 200, 300],
        'l2_leaf_reg': [1, 3, 5],
    }

    best_hitrate = 0
    best_model = None
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        random_search = RandomizedSearchCV(catboost_model, param_grid, scoring='roc_auc', cv=3,
                                           n_iter=20, random_state=42)
        random_search.fit(X_train_fold, y_train_fold)

        y_val_pred_proba = random_search.predict_proba(X_val_fold)[:, 1]
        hitrate = calculate_hitrate(y_val_fold.values, y_val_pred_proba)

        if hitrate > best_hitrate:
            best_hitrate = hitrate
            best_model = random_search.best_estimator_

    return best_model


# Обучаем
catboost_model = train_catboost(X_train, y_train)