# импорт библилотек
import joblib
import os
import logging
import pandas as pd
import random
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import myFunctions

# путь к файлам проекта:
# -> $PROJECT_PATH
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '.')



def pipeline():
    #print('Car Rental Predict Pipeline')

    df_sessions = pd.read_csv(f'{path}/data/dataset/ga_sessions.csv', low_memory=False)
    df_hits = pd.read_csv(f'{path}/data/dataset/ga_hits.csv', low_memory=False)

    # X = df_full.drop('target_action', axis=1)
    # y = df_full['target_action']
    X, y = myFunctions.separation(df_sessions, df_hits)

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear', class_weight='balanced', C=2, verbose=1,
                           fit_intercept=True, max_iter=100),
        RandomForestClassifier(class_weight='balanced', criterion='gini', min_samples_leaf=5,
                               min_samples_split=3, n_estimators=500),
        DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, min_samples_split=3,
                               class_weight='balanced'),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(100, 30), max_iter=100, solver='adam')
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        #print(f'model: {type(model).__name__}, roc_mean: {score.mean():.4f}, roc_std: {score.std():.4f}')
        logging.info(f'model: {type(model).__name__}, roc_mean: {score.mean():.4f}, roc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)

    model_filename = f'{path}/data/models/car_rental_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    #print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')
    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    joblib.dump({
        'model': best_pipe,
        'metadata': {
            'name': 'Car rental predict model',
            'author': 'Alexey Slabinsky',
            'version': 1,
            'date': datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'roc_auc': best_score
        }}, model_filename)


    logging.info(f'Model is saved as {model_filename}')


if __name__ == '__main__':
    pipeline()