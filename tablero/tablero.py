from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import r2_score
import os

# Load the data
APP_HOME = os.getcwd()
puntajes_pacifico = pd.read_csv(os.path.join(APP_HOME, 'exploracion', 'Pacifico_cleaned.csv'))

# Preprocessing and scaling
puntajes_cols = [col for col in puntajes_pacifico.columns if "punt" in col]
X = puntajes_pacifico.drop(puntajes_cols + ["estu_consecutivo"], axis=1)
y = puntajes_pacifico['punt_global']

input_scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = input_scaler.fit_transform(X)

response_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = response_scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train the predictive model
params = {
    "epochs": 1,
    "hidden_layer_size": 50,
    "embedding_layer_size": 20,
    "neuron_activation": "relu",
    "weight_init": "uniform",
    "learning_rate": 0.01
}

model = Sequential()
model.add(Dense(10, activation=params['neuron_activation'],
                kernel_initializer=params['weight_init'], input_shape=(X_train.shape[1],)))
model.add(Dense(200, activation=params['neuron_activation']))
model.add(Dense(1, activation="linear"))

optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_squared_error"])

history = model.fit(X_train, y_train, epochs=params['epochs'], verbose=1)

predictions_scaled = model.predict(X_test)
predictions = response_scaler.inverse_transform(predictions_scaled).flatten()

r2_out_of_sample = r2_score(y_test, predictions) 

# Function to plot predictions vs actuals
def plot_predictions_vs_actuals(predictions, y_test, title="Predicciones vs Valores Reales"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.flatten(), y=predictions, mode='markers', name='Predicciones'))
    fig.add_trace(go.Scatter(x=y_test.flatten(), y=y_test.flatten(), mode='lines', name='Valores Reales'))
    fig.update_layout(title=title, xaxis_title='Valores Reales', yaxis_title='Predicciones')
    return fig

# Function to plot correlations with punt_global
def plot_punt_global_correlations(df):
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    columns_to_exclude = [col for col in numerical_df.columns if col.startswith('punt_') and col != 'punt_global']
    numerical_df = numerical_df.drop(columns=columns_to_exclude)
    correlations = numerical_df.corr()['punt_global'].drop('punt_global').sort_values(ascending=False)
    corr_df = correlations.reset_index()
    corr_df.columns = ['Variable', 'Correlation']
    corr_df['Sign'] = corr_df['Correlation'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    fig = px.bar(corr_df, x='Correlation', y='Variable', orientation='h', color='Sign', 
                 color_discrete_map={'Positive': 'blue', 'Negative': 'red'},
                 title='Pearson Correlation with punt_global',
                 labels={'Correlation': 'Pearson Correlation', 'Variable': 'Variables'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=1000, width=1200)
    return fig

# Function to get correlation data as DataFrame
def get_punt_global_correlations(df):
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    columns_to_exclude = [col for col in numerical_df.columns if col.startswith('punt_') and col != 'punt_global']
    numerical_df = numerical_df.drop(columns=columns_to_exclude)
    correlations = numerical_df.corr()['punt_global'].drop('punt_global').sort_values(ascending=False)
    corr_df = correlations.reset_index()
    corr_df.columns = ['Variable', 'Correlation']
    return corr_df

app = Dash(__name__)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Predicciones vs Valores Reales', value='tab-1'),
        dcc.Tab(label='Correlaciones con punt_global', value='tab-2'),
        dcc.Tab(label='Tabla de Correlaciones', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Graph(figure=plot_predictions_vs_actuals(predictions, y_test)),
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.Graph(figure=plot_punt_global_correlations(puntajes_pacifico)),
        ])
    elif tab == 'tab-3':
        corr_df = get_punt_global_correlations(puntajes_pacifico)
        return html.Div([
            html.H3('Correlaciones con punt_global'),
            dash_table.DataTable(
                data=corr_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in corr_df.columns],
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '150px',
                    'width': '150px',
                    'maxWidth': '150px',
                }
            )
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
