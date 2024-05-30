import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from  sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn
import plotly.express as px
from keras import Sequential
from keras.layers import Dense
from keras import optimizers
import mlflow.keras

APP_HOME=os.getcwd()
os.chdir(APP_HOME)
puntajes_pacifico=pd.read_csv(APP_HOME + '\exploracion\Pacifico_cleaned.csv')

def plot_punt_global_correlations(df=puntajes_pacifico):

    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    columns_to_exclude = [col for col in numerical_df.columns if col.startswith('punt_') and col != 'punt_global']
    numerical_df = numerical_df.drop(columns=columns_to_exclude)
    
    correlations = numerical_df.corr()['punt_global'].drop('punt_global').sort_values(ascending=False)
    
    corr_df = correlations.reset_index()
    corr_df.columns = ['Variable', 'Correlation']
    
    corr_df['Sign'] = corr_df['Correlation'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    
    fig = px.bar(corr_df, x='Correlation', y='Variable', orientation='h',
                 color='Sign', color_discrete_map={'Positive': 'blue', 'Negative': 'red'},
                 title='Pearson Correlation with punt_global',
                 labels={'Correlation': 'Pearson Correlation', 'Variable': 'Variables'})
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=1000, width=1200)
    
    fig.show()


variables_relevantes = [
    'fami_estratovivienda', 'fami_tieneinternet', 'fami_tienecomputador',
    'fami_tieneautomovil', 'madre_Educación profesional completa', 
    'padre_Educación profesional completa', 'fami_tienelavadora', 'punt_global'
]

data = puntajes_pacifico[variables_relevantes]

X = data.drop(['punt_global'], axis=1)
y = data['punt_global']

input_scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = input_scaler.fit_transform(X)

response_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = response_scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

def train_and_log_model(epochs, hidden_layer_size):
    # Define la arquitectura del modelo
    model = Sequential()
    model.add(Dense(40, activation='relu', kernel_initializer='uniform', input_shape=(X_train.shape[1],)))
    model.add(Dense(hidden_layer_size, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compila el modelo
    optimizer = optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

    # Entrena el modelo
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.2)

    # Calcula las predicciones y las métricas
    predictions_scaled = model.predict(X_test)
    predictions = response_scaler.inverse_transform(predictions_scaled).flatten()
    y_test_actual = response_scaler.inverse_transform(y_test).flatten()

    mse = mean_squared_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)

    # Registra el modelo y las métricas con MLflow
    with mlflow.start_run():
        params = {
            'epochs': epochs,
            'hidden_layer_size': hidden_layer_size,
            'learning_rate': 0.01
        }
        mlflow.log_params(params)

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('R2 Score', r2)

        mlflow.keras.log_model(model, 'model')

    return mse, r2

puntajes_pacifico[[_ for _ in puntajes_pacifico.columns if re.search("punt",_)]]
puntajes_pacifico["punt_global"].hist(bins=30)
puntajes_cols=[_ for _ in puntajes_pacifico.columns if re.search("punt",_)]

data=puntajes_pacifico.copy()
X = data.drop( puntajes_cols+["estu_consecutivo"], axis=1)
y = data['punt_global']

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:8000")