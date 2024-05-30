from dash import Dash, dcc, html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
import os

# Cargar los datos
APP_HOME = os.getcwd()
puntajes_pacifico = pd.read_csv(os.path.join(APP_HOME, 'exploracion', 'Pacifico_cleaned.csv'))

# Preprocesamiento y escalado
puntajes_cols = [col for col in puntajes_pacifico.columns if "punt" in col]
X = puntajes_pacifico.drop(puntajes_cols + ["estu_consecutivo"], axis=1)
y = puntajes_pacifico['punt_global']

input_scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = input_scaler.fit_transform(X)

response_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = response_scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Cargar el modelo previamente entrenado
model = load_model(os.path.join(APP_HOME, 'model.h5'))

def plot_predictions_vs_actuals(predictions, y_test, title="Predicciones vs Valores Reales"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicciones'))
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Valores Reales'))
    fig.update_layout(title=title, xaxis_title='Valores Reales', yaxis_title='Predicciones')
    return fig

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

app = Dash(__name__)

sidebar = html.Div(
    [
        html.H2("Análisis de Puntajes"),
        html.Nav(
            [
                html.A("Información General", href="/info", className="nav-link"),
                html.A("Análisis Explicativo", href="/explic", className="nav-link"),
                html.A("Modelo Predictivo", href="/predict", className="nav-link"),
            ],
            className="nav flex-column"
        ),
    ],
    style={"width": "20%", "position": "fixed", "height": "100%", "background-color": "#f8f9fa"}
)

content = html.Div(id="page-content", style={"margin-left": "20%", "padding": "20px"})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content,
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/info":
        return html.Div([
            html.H1("Información General"),
            html.P("Contenido de información general sobre el análisis."),
        ])
    elif pathname == "/explic":
        return html.Div([
            html.H1("Análisis Explicativo"),
            dcc.Graph(id='correlation-graph', figure=plot_punt_global_correlations(puntajes_pacifico)),
        ])
    elif pathname == "/predict":
        return html.Div([
            html.H1("Modelo Predictivo"),
            html.Label("Seleccionar ID del Cliente"),
            dcc.Input(id='customer-id', type='number', value=1),
            html.Div(id='prediction-output'),
        ])
    return "404 Página no encontrada"

@app.callback(
    Output('prediction-output', 'children'),
    [Input('customer-id', 'value')]
)
def update_prediction(customer_id):
    customer_data = X_test[customer_id:customer_id+1]
    prediction = model.predict(customer_data)
    actual = y_test[customer_id]
    prediction = response_scaler.inverse_transform(prediction).flatten()
    actual = response_scaler.inverse_transform(actual.reshape(1, -1)).flatten()
    return html.Div([
        html.H3(f"Predicción: {prediction[0]:.2f}"),
        html.H3(f"Valor Real: {actual[0]:.2f}"),
        dcc.Graph(figure=plot_predictions_vs_actuals(prediction, actual)),
    ])

if __name__ == '__main__':
    app.run_server(debug=True)