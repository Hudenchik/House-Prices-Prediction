import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath: str):
  """Загрузка данных из CSV файла."""
  return pd.read_csv(filepath)


def build_pipeline(num_features: list, cat_features: list) -> Pipeline:
  """Создание Pipeline для предобработки и модели."""
  num_transformer = Pipeline(
      steps=[
          ('imputer', SimpleImputer(strategy='median')),
          ('scaler', StandardScaler()),
      ]
  )

  cat_transformer = Pipeline(
      steps=[
          ('imputer', SimpleImputer(strategy='most_frequent')),
          ('onehot', OneHotEncoder(handle_unknown='ignore')),
      ]
  )

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', num_transformer, num_features),
          ('cat', cat_transformer, cat_features),
      ]
  )

  pipeline = Pipeline(
      steps=[('preprocessor', preprocessor), ('regressor', Ridge())]
  )

  return pipeline


def main():
  # 1. Загрузка
  df = load_data('data/train.csv')

  num_features = [
      'GrLivArea',
      'GarageCars',
      'TotalBsmtSF',
      'FullBath',
      'YearBuilt',
  ]
  cat_features = ['Neighborhood', 'HouseStyle']
  target = 'SalePrice'

  X = df[num_features + cat_features]
  y = df[target]

  # 2. Сплит
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # 3. Обучение
  pipeline = build_pipeline(num_features, cat_features)

  param_grid = {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}
  search = GridSearchCV(
      pipeline, param_grid, cv=5, scoring='neg_mean_squared_error'
  )
  search.fit(X_train, y_train)

  best_model = search.best_estimator_

  # 4. Оценка
  y_pred = best_model.predict(X_test)
  print(f'R² Score: {r2_score(y_test, y_pred):.4f}')
  print(f'RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
  print(f'MAE:      {mean_absolute_error(y_test, y_pred):.2f}')

  # 5. Сохранение обученной модели на диск
  joblib.dump(best_model, 'models/ridge_model.pkl')
  print('Модель успешно сохранена в models/ridge_model.pkl')


if __name__ == '__main__':
  main()