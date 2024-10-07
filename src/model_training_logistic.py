import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib
from scipy.sparse import load_npz

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка подготовленных данных
X_train = load_npz('C:/Users/User/PycharmProjects/MMO/data/processed/X_train.npz')
y_train = pd.read_csv('C:/Users/User/PycharmProjects/MMO/data/processed/y_train.csv')
logging.info("Data loaded successfully.")

# Преобразование y_train в одномерный массив
y_train = y_train.values.ravel()

# Выбор модели
model = LogisticRegression(random_state=42, max_iter=1000)

# Кросс-валидация
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
logging.info(f"Cross-validation scores: {cv_scores}")
logging.info(f"Mean cross-validation score: {cv_scores.mean()}")

# Обучение модели
model.fit(X_train, y_train)
logging.info("Model trained successfully.")

# Сохранение модели
joblib.dump(model, 'C:/Users/User/PycharmProjects/MMO/models/logistic_regression_model.pkl')
logging.info("Model saved successfully.")