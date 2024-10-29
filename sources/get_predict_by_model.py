import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt

from_file = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть

from_file.load_model("catboost_model_final_proj")

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

hitrate = calculate_hitrate(y_test.values, from_file.predict_proba(X_test)[:, 1], k = 5)