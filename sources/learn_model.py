import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve, RocCurveDisplay, auc, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

# Dataset for posts recommendation learning
class Recommend_Data(Dataset):
    def __init__(self, df):

        self.target = df['target']

        self.data = df.drop(['target', 'user_id', 'post_id'], axis=1)

    def __getitem__(self, idx):
        vector = self.data.loc[idx]
        target = self.target.loc[idx]

        vector = torch.FloatTensor(vector)
        target = torch.FloatTensor([target])

        return vector, target

    def __len__(self):
        return len(self.target)

# FC NN for classification
def create_nn_to_classify():
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

    return FirstModel()

# Calculate NN output and binary accuracy based on true results
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# Train cycle of NN
def train(model, train_loader, device, optimizer):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    train_loss = 0
    train_accuracy = 0

    for x, y in tqdm(train_loader, desc='Train'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)

        output, y = output.cpu(), y.cpu()

        loss = loss_fn(output, y)

        train_loss += loss.item()

        train_accuracy += binary_accuracy(output, y)

        loss.backward()

        optimizer.step()

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    train_loss, train_accuracy = train_loss, train_accuracy

    return train_loss, train_accuracy

# Calculate NN output and estimate accuracy
@torch.inference_mode()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0
    total_accuracy = 0

    for x, y in tqdm(loader, desc='Evaluation'):
        x, y = x.to(device), y.to(device)

        output = model(x)

        output, y = output.cpu(), y.cpu()
        loss = loss_fn(output, y)

        total_loss += loss.item()
        total_accuracy += binary_accuracy(output, y)

    total_loss /= len(loader)
    total_accuracy /= len(loader)

    return total_loss, total_accuracy

# Plot graphs of accuraccy and loss change dynamic during epochs
def plot_stats(
        train_loss: list[float],
        valid_loss: list[float],
        train_accuracy: list[float],
        valid_accuracy: list[float],
        title: str
):
    plt.figure(figsize=(16, 8))

    plt.title(title + ' loss')

    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(16, 8))

    plt.title(title + ' accuracy')

    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(valid_accuracy, label='Valid accuracy')
    plt.legend()
    plt.grid()

    plt.show()

# Learn NN for the specified number of epochs
def whole_train_valid_cycle(model,
                            num_epochs,
                            title,
                            train_loader,
                            test_loader,
                            device,
                            optimizer):
    train_loss_history, valid_loss_history = [], []
    train_accuracy_history, valid_accuracy_history = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, device, optimizer)
        valid_loss, valid_accuracy = evaluate(model, test_loader, device)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(valid_accuracy)

        clear_output()

        plot_stats(
            train_loss_history, valid_loss_history,
            train_accuracy_history, valid_accuracy_history,
            title
        )

# Download user's data from the SQL database
def get_user_df():

    # Установка соединения с базой данных
    user = pd.read_sql("SELECT * FROM public.user_data;", os.getenv('DATABASE_URL'))
    print(user.head())
    return user

# Download posts data from the SQL database
def get_post_df():

    # Установка соединения с базой данных
    post = pd.read_sql("SELECT * FROM public.post_text_df;", os.getenv('DATABASE_URL'))
    print(post.head())
    return post

# Download BERT embeddings for posts from the SQL database (prepared in advance)
def get_embedd_df():

    # Загрузка эмбеддингов из файла в Kaggle Inputs
    embedds = pd.read_sql(f"SELECT * FROM {os.getenv('EMBEDD_DF_NAME')};", os.getenv('DATABASE_URL'))
    print(embedds.head())
    return embedds

# Obtaining DF with vector of all features for NN learning
def get_vector_df(feed_n_lines=1024000):

    # Установка соединения с базой данных
    user = get_user_df()
    post = get_post_df()
    embedd = get_embedd_df()
    feed = pd.read_sql(f"SELECT * FROM public.feed_data order by random() LIMIT {feed_n_lines};", os.getenv('DATABASE_URL'))

    feed = feed.drop_duplicates()
    print(feed.head())

    # Поработаем с категориальными колонками для таблицы new_user. Колонку exp_group тоже считаем как категориальную

    new_user = user.drop('city', axis=1)

    categorical_columns = []
    categorical_columns.append('country')
    # categorical_columns.append('os')
    # categorical_columns.append('source')
    categorical_columns.append('exp_group')  # разобью по группам категориальный признак

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

    # готовая таблица User для обучения
    new_user.head()
    num_user_full = new_user['user_id'].nunique()
    print(f'Число уникальных юзеров:{num_user_full}')

    # Длина текста поста - новый признак для таблицы Post
    post['text_length'] = post['text'].apply(len)
    # post = post.rename(columns={"text": "text_feature"})

    # Убираем исходные тексты из признаков
    post = post.drop(['text'], axis=1)

    # Конкатенируем датафрейм Post с фичами-эмбеддингами
    embedd = embedd.add_prefix('embed_')

    # Стандартизация данных - вычитаем среднее значение при помощи скейлера
    scaler = StandardScaler()

    # убрано .to_array(), использую датафрейм
    embedd_scaled = scaler.fit_transform(embedd)

    # Применение PCA
    pca = PCA(n_components=200)

    X_pca = pca.fit_transform(embedd_scaled)
    X_pca = pd.DataFrame(X_pca)
    X_pca = X_pca.add_prefix('PCA_')

    # # Коэффициенты главных компонент
    # components = pca.components_
    # # Создание DataFrame для значимости
    # importance_df = pd.DataFrame(components, columns=embedd.columns)

    # #print(importance_df.head())

    # # Суммирование абсолютных значений коэффициентов для каждой колонки
    # importance_scores = importance_df.abs().sum(axis=0)

    # # Сортировка по значимости
    # sorted_importance = importance_scores.sort_values(ascending=False)

    # print(sorted_importance.head(10))

    # # Получение топ имен колонок, отсортированных по значимости
    # top_features = sorted_importance[:300].index.tolist()

    # Конкатенирую с топом эмбеддингов, максимально вносящих вклад в PCA фичи
    post = pd.concat([post, X_pca], axis=1, join='inner')
    embedd_columns = embedd.columns

    print(post.head())

    # разобью по группам категориальный признак из Post
    categorical_columns.append('topic')

    # Выбираем только числовые столбцы таблицы Post для преобразования
    numeric_columns = post.select_dtypes(include=['float64', 'int64']).columns

    # Преобразуем только числовые столбцы в float32
    post[numeric_columns] = post[numeric_columns].astype('float32')

    # Выбираем только числовые столбцы таблицы Feed для преобразования
    numeric_columns = feed.select_dtypes(include=['float64', 'int64']).columns

    # Преобразуем только числовые столбцы в float32
    feed[numeric_columns] = feed[numeric_columns].astype('float32')

    # Ренейм action на случай пересечений с колонками TD-IDF
    feed = feed.rename(columns={"action": "action_class"})

    # Теперь нужно объединить все с таблицей Feed, чтобы получить мастер-таблицу для трейна

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
    # df['views_per_post'] = df.groupby('post_id')['action_class'].apply(lambda x: 1 if x == 0 else 0).transform('sum')
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
    df['time_indicator'] = (df['year'] - 2021) * 360 * 24 + df['month'] * 30 * 24 + df['day'] * 24 + df['hour']

    categorical_columns.append('month')  # разобью по группам month  из Feed

    # Генерим фичи: топ topic для пользователей из feed по лайкам/просмотрам
    main_liked_topics = df[df['action_class'] == 1].groupby(['user_id'])['topic'].agg(
        lambda x: np.random.choice(x.mode())).to_frame().reset_index()
    main_liked_topics = main_liked_topics.rename(columns={"topic": "main_topic_liked"})
    main_viewed_topics = df[df['action_class'] == 0].groupby(['user_id'])['topic'].agg(
        lambda x: np.random.choice(x.mode())).to_frame().reset_index()
    main_viewed_topics = main_viewed_topics.rename(columns={"topic": "main_topic_viewed"})

    # Присоединяем к мастер-таблице
    df = pd.merge(df, main_liked_topics, on='user_id', how='left')
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
    # df['views_per_user'] = df.groupby('user_id')['action_class'].apply(lambda x: 1 if x == 0 else 0).transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
    df['views_per_user'] = df.groupby('user_id')['action_class'].transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

    # Присоединяем к мастер-таблице
    df = pd.merge(df, likes_per_user, on='user_id', how='left')

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

    print(categorical_columns)

    # One-hot encoding для всех категориальных колонок
    for col in categorical_columns:
        one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype='int32')

        df = pd.concat((df.drop(col, axis=1), one_hot), axis=1)

    # Оставляю user_id и post_id, их нужно будет дропнуть при создании датасета
    # df = df.drop(['user_id', 'post_id'], axis=1)

    df = df.astype('float32')
    print('Итоговый датасет:')
    print(df.head)

    # сохраняю с user_id для генерации таблицы признаков для сервера
    # df.to_csv('df_to_learn.csv', sep=';', index=False)

    # Получение общего объема памяти, занимаемой DataFrame
    total_memory = df.memory_usage(deep=True).sum()
    print(f"\nОбщий объем памяти, занимаемой DataFrame: {total_memory} байт")
    print(df.dtypes)

    return df, categorical_columns, post


def learn_model(df_size=1024000, n_epochs=20):

    # Наберем необходимые записи
    data, cat_columns, post = get_vector_df(feed_n_lines=df_size)

    # Dataset init from class
    dataset = Recommend_Data(data)

    # Datasest random split, 20% for test
    train_dataset, test_dataset = random_split(dataset,
                                               (int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)))

    # 64 batch size - optimal (empyrical)
    train_loader = DataLoader(train_dataset, batch_size=64, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True)

    # Choosing device: Cuds if presented unless CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    # Creating NN for learning
    model = create_nn_to_classify()
    model = model.to(device)

    # Adam optimizer with default learning rate
    optimizer = Adam(model.parameters(), lr=1e-3)

    whole_train_valid_cycle(model,
                            n_epochs,
                            'Learning post recommendations by likes',
                            train_loader,
                            test_loader,
                            device,
                            optimizer)

    torch.save(model.state_dict(), '/nn_estinmate_likes_200xPCA_embedds_1024k_drop_03_02.pt')


    # Estimate accuracy, F-measure and AUC based on the whole dataseb

    X = data.drop(['target', 'user_id', 'post_id'], axis=1)
    y = data.target
    model.eval().cpu()

    X_tens = torch.FloatTensor(X.values)
    F_X = torch.round(torch.sigmoid(model(X_tens))).detach().numpy().astype("float32")
    Prob_X = torch.sigmoid(model(X_tens)).detach().numpy().astype("float32")

    # F-measure
    f1_loc_tr = round(f1_score(y,
                               F_X,
                               average='weighted'), 5)
    print(f'F-measure for FC NN: {f1_loc_tr}')

    # AUC
    fpr, tpr, thd = roc_curve(y, Prob_X)
    print(f'AUC for FC NN: {auc(fpr, tpr):.5f}')

    acc = accuracy_score(y, F_X)
    print(f'Accuracy for FC NN: {acc:.5f}')

    # ROC curve
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    return data, post, model
