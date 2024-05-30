import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn
import mlflow.keras
import mlflow
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

APP_HOME=os.getcwd()
os.chdir(APP_HOME)
puntajes_pacifico=pd.read_csv(APP_HOME + '\exploracion\Pacifico_cleaned.csv')
puntajes_pacifico[[_ for _ in puntajes_pacifico.columns if re.search("punt",_)]]
puntajes_cols=[_ for _ in puntajes_pacifico.columns if re.search("punt",_)]
data=puntajes_pacifico.copy()
X = data.drop( puntajes_cols+["estu_consecutivo"], axis=1)
y = data['punt_global']
#mlflow.set_tracking_uri("http://127.0.0.1:8000")


'''
MODELO PREDICTIVO
'''

params = {
    "epochs": 1,

    "hidden_layer_size": 50,
    "embedding_layer_size": 20,
    
    "neuron_activation": "relu",
    "weight_init": "uniform",
    "learning_rate": 0.01
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled_train = input_scaler.fit_transform(X_train)
x_scaled_test = input_scaler.transform(X_test)

y_train = y_train.to_numpy().reshape(-1, 1)  
response_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled_train = response_scaler.fit_transform(y_train)

model = Sequential()
model.add(Dense(10, activation=params['neuron_activation'],
                kernel_initializer=params['weight_init'], input_shape=(x_scaled_train.shape[1],)))
model.add(Dense(200, activation=params['neuron_activation']))
model.add(Dense(1, activation="linear"))

optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_squared_error"])

history = model.fit(x_scaled_train, y_scaled_train, epochs=1, verbose=1)

predictions_scaled = model.predict(x_scaled_test)
predictions = response_scaler.inverse_transform(predictions_scaled).flatten()

r2_out_of_sample = r2_score(y_test, predictions) 
mlflow.log_metric("R2_out_of_sample", r2_out_of_sample)

mlflow.keras.log_model(model, "model")

mlflow.end_run()

try:
    X_test=X_test.drop("predictions", axis=1)
except:
    pass


best_model=model
best_model.summary()

x_scaled_test = input_scaler.transform(X_test)
predictions_scaled = model.predict(x_scaled_test)
predictions = response_scaler.inverse_transform(predictions_scaled).flatten()
pd.Series(predictions).describe()

import pandas as pd
import plotly.graph_objects as go

X_test["predictions"] = predictions
grouped_predictions = X_test.groupby("fami_estratovivienda")["predictions"].mean().reset_index()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=grouped_predictions['fami_estratovivienda'],
    y=grouped_predictions['predictions'],
    marker=dict(color='blue', line=dict(color='white', width=2)),
    width=0.3
))

for index, row in grouped_predictions.iterrows():
    fig.add_annotation(
        x=row['fami_estratovivienda'],
        y=row['predictions'],
        text=f"{row['predictions']:.3f}",
        showarrow=False,
        yshift=10
    )

fig.update_layout(
    xaxis_title="Estrato de Vivienda",
    yaxis_title="Average Prediction",
    title="Average Model Predictions by Estrato de Vivienda",
    xaxis={'categoryorder':'total descending'}
)

fig.show()

try:
    X_test=X_test.drop("predictions", axis=1)
except:
    pass


'''
MODELO DESCRIPTIVO
'''
import pandas as pd
import plotly.express as px

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

plot_punt_global_correlations()


def print_punt_global_correlations(df=puntajes_pacifico):
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    columns_to_exclude = [col for col in numerical_df.columns if col.startswith('punt_') and col != 'punt_global']
    numerical_df = numerical_df.drop(columns=columns_to_exclude)
    
    correlations = numerical_df.corr()['punt_global'].drop('punt_global').sort_values(ascending=False)
    
    print("Correlaciones con punt_global:")
    print(correlations)

print_punt_global_correlations()

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
