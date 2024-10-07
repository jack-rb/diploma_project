import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import joblib
from scipy.sparse import load_npz

# Загрузка модели
model = joblib.load('C:/Users/User/PycharmProjects/MMO/models/random_forest_model_CV.pkl')

# Загрузка тестовых данных
X_test = load_npz('C:/Users/User/PycharmProjects/MMO/data/processed/X_test.npz')
y_test = pd.read_csv('C:/Users/User/PycharmProjects/MMO/data/processed/y_test.csv')

# Преобразование y_test в одномерный массив
y_test = y_test.values.ravel()

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Оценка модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))