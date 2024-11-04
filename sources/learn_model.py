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
import os, pickle

# Подготовка фичей по данным от сервера
def get_features_df(feed_n_lines=1000000):

    # Установка соединения с базой данных
    user = get_user_df()
    post = get_post_df()
    feed = pd.read_sql(f"SELECT * FROM public.feed_data order by random() LIMIT {feed_n_lines};", os.getenv('DATABASE_URL'))
    feed = feed.drop_duplicates()
    print(feed.head())

    # Поработаем с категориальными колонками для таблицы new_user. Колонку exp_group тоже считаем как категориальную

    new_user = user.drop('city', axis=1)

    categorical_columns = []
    categorical_columns.append('country')
    #categorical_columns.append('os')
    #categorical_columns.append('source')
    categorical_columns.append('exp_group')  # разобью по группам категориальный признак

    #print(categorical_columns)

    # for col in categorical_columns:
    #     one_hot = pd.get_dummies(new_user[col], prefix=col, drop_first=True, dtype='int64')
    #
    #     new_user = pd.concat((new_user.drop(col, axis=1), one_hot), axis=1)

    # Добавил булевый признак по главным городам в представленных странах, остальных городов слишком много
    capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
                'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
    cap_bool = user.city.apply(lambda x: 1 if x in capitals else 0)

    # добавил признак по главным городам в представленных странах
    new_user = pd.concat([new_user, cap_bool], axis=1, join='inner')
    new_user = new_user.rename(columns={"city": "city_capital"})

    # Выбираем только числовые столбцы для преобразования
    numeric_columns = new_user.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns

    # Преобразуем только числовые столбцы в float32
    new_user[numeric_columns] = new_user[numeric_columns].astype('float32')

    #готовая таблица для обучения
    new_user.head()
    num_user_full = new_user['user_id'].nunique()
    print(f'Число уникальных юзеров:{num_user_full}')

    # Теперь обработаю текстовую колонку при помощи TF-IDF. Текстов много, тем - тоже, потому выделю две фичи:
    # по 5 ключевым словам выдам медианный TF-IDF и максимальный, плюс средний показатель по 10 словам в топе.
    # Буду считать их за маркеры аутентичности статьи. В дальнейшем надо выявить предпочтения юзеров: как много лайкают
    # вообще и в топе - по какой теме.

    num_post_full = post['post_id'].nunique()
    print(f'Число уникальных постов:{num_post_full}')

    # Зафиттим наши данные в TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english', strip_accents='unicode', min_df = 0.01, max_features = 2000)
    tfidf_matrix = tfidf.fit_transform(post['text'].fillna('unknown'))
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.reset_index(drop=True, inplace=True)
    post.reset_index(drop=True, inplace=True)

    # Выделяем главные значаящие слова по методу PCA - от максимальной дисперсии
    # Далее берем топ значимых TF-IDF для PCA фичей, и по ним делаем кластеризацию
    # Это будет новый категориальный признак

    # Стандартизация данных - вычитаем среднее значение при помощи скейлера
    scaler = StandardScaler()

    # убрано .to_array(), использую датафрейм
    X_scaled = scaler.fit_transform(tfidf_df)

    # Применение PCA
    pca = PCA(n_components=200)

    X_pca = pca.fit_transform(X_scaled)
    X_pca = pd.DataFrame(X_pca)
    X_pca = X_pca.add_prefix('PCA_')

    # Коэффициенты главных компонент
    components = pca.components_

    # Создание DataFrame для значимости
    importance_df = pd.DataFrame(tfidf_df, columns=tfidf.get_feature_names_out())

    # Суммирование абсолютных значений коэффициентов для каждой колонки
    importance_scores = importance_df.abs().sum(axis=0)

    # Сортировка по значимости
    sorted_importance = importance_scores.sort_values(ascending=False)

    # Получение топ имен колонок, отсортированных по значимости
    top_features = sorted_importance[:1500].index.tolist()

    print("Наиболее значимые слова:", top_features)

    # Примени кластеризацию DBSCAN - т.к. не занем сколько и каких кластеров есть
    #clustering = DBSCAN(eps=0.6, min_samples=4).fit_predict(tfidf_df[top_features])
    clustering = KMeans(n_clusters=5, random_state=42, max_iter=1000).fit_predict(tfidf_df[top_features])
    clusters = pd.DataFrame({"clusters": clustering})

    print(clusters.nunique())

    # categorical_columns.append('clusters')

    one_hot = pd.get_dummies(clusters['clusters'], prefix='cluster', drop_first=True, dtype='int32')

    post = pd.concat((post, one_hot), axis=1)

    # # Сократим топ имен колонок, отсортированных по значимости
    # top_features = sorted_importance[:50].index.tolist()
    #
    # # #Присоединим к таблице постов новые признаки от TF-IDF по ключевым словам
    # post = pd.concat([post, tfidf_df[top_features]], axis=1)

    # # Вещенственные метрики для индикации аутентичности поста, по TF-IDF ключевых слов

    # post['tf_idf_median_5'] = 0
    # post['tf_idf_mean_20'] = 0

    # for index, value in post['text'].items():  # смотрим tf-idf только по фильмам

    #     words = pd.DataFrame(tfidf_matrix[index].T.todense(),
    #                          index=feature_names,
    #                          columns=['tfidf'])

    #     words_sort = words.sort_values('tfidf', ascending=False).head(20)
    #     post['tf_idf_median_5'].iloc[index] = words_sort.head(5).median()
    #     post['tf_idf_mean_20'].iloc[index] = words_sort.mean()

    # print(post['tf_idf_median_5'].head(30))

    # Длина текста поста - новый признак
    post['text_length'] = post['text'].apply(len)
    # post = post.rename(columns={"text": "text_feature"})

    # Убираем исходные тексты из признаков
    post = post.drop(['text'], axis=1)

    # разобью по группам категориальный признак из Post
    categorical_columns.append('topic')

    # Выбираем только числовые столбцы таблицы Post для преобразования
    numeric_columns = post.select_dtypes(include=['float64', 'int64']).columns

    # Преобразуем только числовые столбцы в float32
    post[numeric_columns] = post[numeric_columns].astype('float32')

    # Выбираем только числовые столбцы таблицы Feed для преобразования
    numeric_columns = feed.select_dtypes(include=['float64', 'int64' ]).columns

    # Преобразуем только числовые столбцы в float32
    feed[numeric_columns] = feed[numeric_columns].astype('float32')

    # Ренейм action на случай пересечений с колонками TD-IDF
    feed = feed.rename(columns={"action": "action_class"})

    #Теперь нужно объединить все с таблицей Feed, чтобы получить мастер-таблицу для трейна

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

    # Признак-счетчик лайков для постов
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)
    df['post_likes'] = df.groupby('post_id')['action_class'].transform('sum')

    # Признак-счетчик просмотров для постов
    #df['views_per_post'] = df.groupby('post_id')['action_class'].apply(lambda x: 1 if x == 0 else 0).transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
    df['post_views'] = df.groupby('post_id')['action_class'].transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

    # Поправим Datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Нужно отсортировать по time для валидации, по возрастанию
    df = df.sort_values('timestamp')

    # Набираю признаки из timestamp
    df['day_of_week'] = df.timestamp.dt.dayofweek
    df['hour'] = df.timestamp.dt.hour
    df['month'] = df.timestamp.dt.month
    df['day'] = df.timestamp.dt.day
    df['year'] = df.timestamp.dt.year

    # Фича-индикатор суммарного времени с 2021 года до текущего момента просмотра, в часах
    df['time_indicator'] = (df['year'] - 2021)*360*24 + df['month']*30*24 + df['day']*24 + df['hour']

    categorical_columns.append('month')  # разобью по группам month  из Feed
    #categorical_columns.append('year')  # разобью по группам year из Feed

    # OneHotEncoding по month
    # one_hot = pd.get_dummies(df['month'], prefix='month', drop_first=True, dtype='int32')
    #
    # df = pd.concat((df.drop('month', axis=1), one_hot), axis=1)

    # Генерим фичи: топ topic для пользователей из feed по лайкам/просмотрам
    main_liked_topics = df[df['action_class'] == 1].groupby(['user_id'])['topic'].agg(lambda x: np.random.choice(x.mode())).to_frame().reset_index()
    main_liked_topics = main_liked_topics.rename(columns={"topic": "main_topic_liked"})
    main_viewed_topics = df[df['action_class'] == 0].groupby(['user_id'])['topic'].agg(lambda x: np.random.choice(x.mode())).to_frame().reset_index()
    main_viewed_topics = main_viewed_topics.rename(columns={"topic": "main_topic_viewed"})

    # Присоединяем к мастер-таблице
    df = pd.merge(df, main_liked_topics,  on='user_id', how='left')
    df = pd.merge(df, main_viewed_topics, on='user_id', how='left')

    # Заполняем пропуски самой частой категорией
    df['main_topic_liked'].fillna(df['main_topic_liked'].mode().item(), inplace=True)
    df['main_topic_viewed'].fillna(df['main_topic_viewed'].mode().item(), inplace=True)

    # Разобью по группам категориальный признак из Feed
    categorical_columns.append('main_topic_viewed')
    categorical_columns.append('main_topic_liked')

    # Признак-счетчик лайков по юзерам
    likes_per_user = df.groupby(['user_id'])['action_class'].agg(pd.Series.sum).to_frame().reset_index()
    likes_per_user = likes_per_user.rename(columns={"action_class": "likes_per_user"})

    # Признак-счетчик просмотров для юзеров
    #df['views_per_user'] = df.groupby('user_id')['action_class'].apply(lambda x: 1 if x == 0 else 0).transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
    df['views_per_user'] = df.groupby('user_id')['action_class'].transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

    # Присоединяем к мастер-таблице
    df = pd.merge(df, likes_per_user,  on='user_id',how='left')

    # Выбираем только числовые столбцы для преобразования
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    df[numeric_columns] = df[numeric_columns].astype('float32')

    num_user_df = df['user_id'].nunique()
    print(f'Число уникальных юзеров в итоговом датасете:{num_user_df}')

    num_post_df = df['post_id'].nunique()
    print(f'Число уникальных постов в итоговом датасете:{num_post_df}')

    # В датасете есть повторения, где у таргета и action_class несогласованны данные.
    # При этом это одна и та же запись, по сути. Не буду убирать дублеры.
    # Просто задам таргету 1 если у строки был лайк. Тем самым данные не будут противоречить друг другу
    df['target'] = df['target'].astype('int32')
    df['action_class'] = df['action_class'].astype('int32')
    df['target'] = df['target'] | df['action_class']

    # Уберем лишние признаки
    df = df.drop(['timestamp', 'action_class', 'os', 'source', 'day_of_week', 'year'], axis=1)

    # Преобразуем численные категориальные в int32
    df[['exp_group', 'month']] = df[['exp_group', 'month']].astype('int32')

    # сохраняю с user_id для генерации таблицы признаков для сервера
    df.to_csv('df_to_learn.csv', sep=';', index=False)

    # Удаляю ненужный более user_id и post_id
    df = df.drop(['user_id', 'post_id'], axis=1)

    # Получение общего объема памяти, занимаемой DataFrame
    total_memory = df.memory_usage(deep=True).sum()
    print(f"\nОбщий объем памяти, занимаемой DataFrame: {total_memory} байт")
    print(df.dtypes)

    return df, categorical_columns

#Глобальный Hitrate на тесте для оценки финального качества
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
def learn_model(df_size = 2000000):

    # Наберем необходимые записи
    data, cat_columns = get_features_df(feed_n_lines=df_size)

    X = data.drop('target', axis=1)
    y = data.target

    # Разделение данных на обучающую и тестовую выборки без перемешивания
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    search = CatBoostClassifier(verbose=False,
                                depth=6,
                                learning_rate=0.2,
                                iterations=200,
                                l2_leaf_reg=2000,
                                cat_features=cat_columns)

    search.fit(X_train, y_train)

    # Сохраняю модель
    search.save_model('catboost_model_final_proj', format="cbm")

    # Строю распределение фичей по их важности для классификации
    feature_imp = search.feature_importances_

    forest_importances = pd.Series(feature_imp, index=X.columns)
    fig, ax = plt.subplots()
    forest_importances.plot.bar()
    ax.set_title("Feature importances Catboost")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # F-меры на трейне и тесте
    f1_loc_tr = round(f1_score(y_train, search.predict(X_train), average='weighted'), 5)
    f1_loc_test = round(f1_score(y_test, search.predict(X_test), average='weighted'), 5)
    print(f'F-мера для бустинга на трейне: {f1_loc_tr}')
    print(f'F-мера для бустинга на тесте: {f1_loc_test}')

    # AUC на трейне
    fpr, tpr, thd = roc_curve(y_train, search.predict_proba(X_train)[:, 1])
    print(f'AUC для CatBoost на трейне: {auc(fpr, tpr):.5f}')

    # AUC на тесте
    fpr, tpr, thd = roc_curve(y_test, search.predict_proba(X_test)[:, 1])
    print(f'AUC для CatBoost на тесте: {auc(fpr, tpr):.5f}')

    hitrate = calculate_hitrate(y_test.values, search.predict_proba(X_test)[:, 1], k = 5)

    print(f'Hitrate для бустинга на тесте: {hitrate}')

    # ROC кривая для теста
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()

    return search, data, cat_columns
