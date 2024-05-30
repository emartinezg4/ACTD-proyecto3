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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the predictive model
params = {
    "epochs": 1,
    "hidden_layer_size": 10,
    "embedding_layer_size": 20,
    "neuron_activation": "relu",
    "weight_init": "uniform",
    "learning_rate": 0.01
}

model = Sequential()
model.add(Dense(20, activation=params['neuron_activation'],
                kernel_initializer=params['weight_init'], input_shape=(X_train.shape[1],)))
model.add(Dense(150, activation=params['neuron_activation']))
model.add(Dense(1, activation="linear"))

optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_squared_error"])

history = model.fit(X_train, y_train, epochs=params['epochs'], verbose=1)

# Predetermined values for the first tab
predetermined_values = {
    1: 246.364,
    2: 257.610,
    3: 267.194,
    4: 282.209,
    5: 287.356,
    6: 302.773
}

def plot_average_predictions_by_estrato(predetermined_values):
    df = pd.DataFrame(list(predetermined_values.items()), columns=['Estrato de Vivienda', 'Predicción Promedio'])
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['Estrato de Vivienda'],
        y=df['Predicción Promedio'],
        marker=dict(color='blue', line=dict(color='white', width=2)),
        width=0.3
    ))

    for index, row in df.iterrows():
        fig.add_annotation(
            x=row['Estrato de Vivienda'],
            y=row['Predicción Promedio'],
            text=f"{row['Predicción Promedio']:.3f}",
            showarrow=False,
            yshift=10
        )

    fig.update_layout(
        xaxis_title="Estrato de Vivienda",
        yaxis_title="Predicción Promedio del Puntaje Global",
        title="Predicción del Puntaje Global Promedio Según Estrato de Vivienda",
        xaxis={'categoryorder':'total descending'}
    )
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
                 title='Pearson Correlation con Puntaje Global',
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
    corr_df.columns = ['Variable', 'Pearson Correlation']
    return corr_df

app = Dash(__name__)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Predicciones', value='tab-1'),
        dcc.Tab(label='Correlaciones de Pearson', value='tab-2'),
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
            dcc.Graph(figure=plot_average_predictions_by_estrato(predetermined_values)),
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.Graph(figure=plot_punt_global_correlations(puntajes_pacifico)),
        ])
    elif tab == 'tab-3':
        corr_df = get_punt_global_correlations(puntajes_pacifico)
        return html.Div([
            html.H3('Correlaciones con Puntaje Global'),
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
