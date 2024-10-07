import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from scipy.sparse import save_npz, load_npz

# Загрузка данных
data = pd.read_csv('C:/Users/User/PycharmProjects/MMO/data/raw/data.csv')

# Очистка данных
data = data.drop_duplicates()
data = data.dropna()

# Разделение признаков и меток
X = data.drop('label', axis=1)
y = data['label']

# Предобработка числовых и категориальных признаков
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = preprocessor.fit_transform(X)

# Балансировка классов
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сохранение подготовленных данных
save_npz('C:/Users/User/PycharmProjects/MMO/data/processed/X_train.npz', X_train)
save_npz('C:/Users/User/PycharmProjects/MMO/data/processed/X_test.npz', X_test)
y_train.to_csv('C:/Users/User/PycharmProjects/MMO/data/processed/y_train.csv', index=False)
y_test.to_csv('C:/Users/User/PycharmProjects/MMO/data/processed/y_test.csv', index=False)