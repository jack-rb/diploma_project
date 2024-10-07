import pandas as pd
import joblib

# Загрузка модели
model = joblib.load('../models/random_forest_model.pkl')

# Загрузка новых данных
new_data = pd.read_csv('../data/raw/new_data.csv')

# Предобработка новых данных (предполагается, что preprocessor уже создан ранее)
preprocessor = joblib.load('../models/preprocessor.pkl')
new_data_processed = preprocessor.transform(new_data)

# Предсказание на новых данных
predictions = model.predict(new_data_processed)

# Сохранение предсказаний
pd.DataFrame(predictions, columns=['predictions']).to_csv('../data/processed/predictions.csv', index=False)