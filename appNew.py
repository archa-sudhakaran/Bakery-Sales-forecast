import dash
import dash_auth
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import numpy as np
import io
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# Initialize app
app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(
    app, {
        'admin': 'admin123',
        'adminnew': 'admintwo'
    }
)

# Load dataset
df = pd.read_csv("C:/Users/User/DashProject\pythonProject11/datasetbakery.csv")

# Preprocessing
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df['sales_qntty'] = pd.to_numeric(df['sales_qntty'], errors='coerce')
df.fillna(0, inplace=True)

# Custom colors
custom_colors = {
    "dessert": "#FF5733",
    "cake": "#6A0572",
    "sweets": "#28A745",
    "pastry": "#FFC107",
    "cookie": "#007BFF"
}

# Categories
categories = df['category'].dropna().unique()
category_options = [{'label': 'All', 'value': 'All'}] + [{'label': cat, 'value': cat} for cat in categories]

# Forecasting Models
model_options = [
    {'label': 'ARIMA', 'value': 'ARIMA'},
    {'label': 'Prophet', 'value': 'Prophet'}
]

# ARIMA Prediction Function
def predict_sales_arima(data, future_periods=30):
    if data.empty:
        return pd.DataFrame(columns=['date', 'sales_qntty', 'predicted_sales', 'type']), None

    sales_df = data.groupby('date')['sales_qntty'].sum().reset_index()
    full_date_range = pd.date_range(start=sales_df['date'].min(), end=sales_df['date'].max(), freq='D')
    sales_df = sales_df.set_index('date').reindex(full_date_range, fill_value=0).reset_index()
    sales_df.rename(columns={'index': 'date'}, inplace=True)

    try:
        model = sm.tsa.ARIMA(sales_df['sales_qntty'], order=(2, 1, 2))
        model_fit = model.fit()
        sales_df['predicted_sales'] = model_fit.fittedvalues
        sales_df['type'] = "Past Sales"

        future_dates = pd.date_range(start=sales_df['date'].max() + pd.Timedelta(days=1), periods=future_periods, freq='D')
        future_sales = model_fit.forecast(steps=future_periods)
        future_df = pd.DataFrame({
            'date': future_dates,
            'sales_qntty': np.nan,
            'predicted_sales': future_sales,
            'type': "Future Sales"
        })
        sales_df = pd.concat([sales_df, future_df], ignore_index=True)

        # Calculate Accuracy
        past_sales = sales_df[sales_df['type'] == 'Past Sales']['sales_qntty']
        past_predictions = sales_df[sales_df['type'] == 'Past Sales']['predicted_sales']
        rmse = np.sqrt(mean_squared_error(past_sales, past_predictions))
        mae = mean_absolute_error(past_sales, past_predictions)
        accuracy_metrics = {'RMSE': rmse, 'MAE': mae}

    except Exception as e:
        print(f"ARIMA failed: {e}")

        sales_df['predicted_sales'] = 0
        accuracy_metrics = None

    return sales_df, accuracy_metrics

# Prophet prediction function
def predict_sales_prophet(data, future_periods=30):
    if data.empty:
        return pd.DataFrame(columns=['ds', 'yhat', 'type']), None

    sales_df = data.groupby('date')['sales_qntty'].sum().reset_index()
    sales_df.columns = ['ds', 'y']

    try:
        model = Prophet()
        model.fit(sales_df)
        future = model.make_future_dataframe(periods=future_periods)
        forecast = model.predict(future)
        forecast['type'] = np.where(forecast['ds'] <= sales_df['ds'].max(), 'Past Sales', 'Future Sales')
        forecast.rename(columns={'ds': 'date', 'yhat': 'predicted_sales'}, inplace=True)
        forecast['sales_qntty'] = sales_df['y'].tolist() + [np.nan] * future_periods

        # Calculate Accuracy
        past_sales = forecast[forecast['type'] == 'Past Sales']['sales_qntty']
        past_predictions = forecast[forecast['type'] == 'Past Sales']['predicted_sales']
        # rmse = mean_squared_error(past_sales, past_predictions, squared=False)
        rmse = np.sqrt(mean_squared_error(past_sales, past_predictions))

        mae = mean_absolute_error(past_sales, past_predictions)
        accuracy_metrics = {'RMSE': rmse, 'MAE': mae}

    except Exception as e:
        print(f"Prophet failed: {e}")
        forecast = pd.DataFrame(columns=['date', 'predicted_sales', 'type'])
        accuracy_metrics = None

    return forecast, accuracy_metrics

# Layout
app.layout = html.Div(style={'backgroundColor': '#87CEEB', 'padding': '20px'}, children=[
    html.H1("Sales Analysis Dashboard", style={'text-align': 'center', 'color': 'white'}),

    html.Div([
        html.Label("Select Date Range:", style={'color': 'white'}),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed=df['date'].min().date(),
            max_date_allowed=df['date'].max().date(),
            start_date=df['date'].min().date(),
            end_date=df['date'].max().date(),
            style={'color': 'black'}
        ),
        html.Br(),
        html.Label("Select Category:", style={'color': 'white'}),
        dcc.Dropdown(
            id='category-dropdown',
            options=category_options,
            value='All',
            clearable=False,
            style={'width': '50%'}
        ),
        html.Br(),
        html.Label("Select Forecasting Model:", style={'color': 'white'}),
        dcc.Dropdown(
            id='model-dropdown',
            options=model_options,
            value='ARIMA',
            clearable=False,
            style={'width': '50%'}
        ),
    ], style={'text-align': 'center'}),

    html.Div([
        dcc.Graph(id="sales-bar-chart"),
        dcc.Graph(id="prediction-bar-chart"),
        dcc.Graph(id="category-pie-chart"),
    ]),

    html.Div([
        dcc.Graph(id="forecast-chart"),
        html.Div(id="accuracy-metrics", style={'color': 'white'})
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

    html.Div([
        dash_table.DataTable(
            id='sales-table',
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'rgb(220, 220, 220)',
                'fontWeight': 'bold'
            }
        ),
    ]),

    html.Div([
        html.Button("Download PDF Summary", id="download-pdf-btn", n_clicks=0, style={'margin-top': '20px'}),
        dcc.Download(id="download-pdf")
    ], style={'text-align': 'center', 'margin-top': '20px'}),
])

# Update Graphs Callback
@app.callback(
    [Output("sales-bar-chart", "figure"),
     Output("prediction-bar-chart", "figure"),
     Output("category-pie-chart", "figure"),
     Output("forecast-chart", "figure"),
     Output('sales-table', 'data'),
     Output('accuracy-metrics', 'children')],
    [Input("date-picker", "start_date"),
     Input("date-picker", "end_date"),
     Input("category-dropdown", "value"),
     Input("model-dropdown", "value")]
)
def update_graphs(start_date, end_date, selected_category, selected_model):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if selected_category == "All":
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    else:
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['category'] == selected_category)].copy()

    if filtered_df.empty:
        return px.bar(), px.bar(), px.pie(), px.line(), [], ""

    if selected_model == 'ARIMA':
        forecast_df, accuracy = predict_sales_arima(filtered_df, future_periods=30)
    elif selected_model == 'Prophet':
        forecast_df, accuracy = predict_sales_prophet(filtered_df, future_periods=30)

    past_forecast_df = forecast_df[forecast_df['type'] == "Past Sales"][['date', 'predicted_sales']]
    filtered_df = filtered_df.merge(past_forecast_df, on='date', how='left')

    fig_sales = px.bar(filtered_df, x="name", y="sales_qntty", title="Actual Sales",
                       color="category", color_discrete_map=custom_colors,
                       labels={"sales_qntty": "Sales (Units)"})

    fig_predictions = px.bar(filtered_df, x="name", y="predicted_sales", title="Predicted Sales",
                             color="category", color_discrete_map=custom_colors,
                             labels={"predicted_sales": "Predicted Sales (Units)"})

    fig_category_pie = px.pie(filtered_df, names="category", values="sales_qntty",
                              title="Sales by Category", color="category",
                              color_discrete_map=custom_colors)

    forecast_fig = px.line(
        forecast_df,
        x="date",
        y="predicted_sales",
        title="Sales Forecast",
        labels={"predicted_sales": "Sales Units", "date": "Date", "type": "Sales Type"},
        color="type",
        color_discrete_map={"Past Sales": "blue", "Future Sales": "red"}
    )
    data = filtered_df.to_dict('records')

    if accuracy:
        accuracy_text = f"Model Accuracy: RMSE: {accuracy['RMSE']:.2f}, MAE: {accuracy['MAE']:.2f}"
    else:
        accuracy_text = "Model accuracy could not be calculated."

    return fig_sales, fig_predictions, fig_category_pie, forecast_fig, data, accuracy_text

# Download PDF Callback
@app.callback(
    Output("download-pdf", "data"),
    Input("download-pdf-btn", "n_clicks"),
    State("date-picker", "start_date"),
    State("date-picker", "end_date"),
    State("category-dropdown", "value"),
    prevent_initial_call=True
)
def generate_pdf(n_clicks, start_date, end_date, selected_category):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['category'] == selected_category]

    forecast_df, _ = predict_sales_arima(filtered_df)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Sales Report Summary", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Date Range: {start_date.date()} to {end_date.date()}", styles["Normal"]))
    elements.append(Paragraph(f"Category: {selected_category}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    def build_table(data, col_names):
        table_data = [col_names] + data
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    # Table 1: Top Selling Products
    top_items = filtered_df.groupby('name')['sales_qntty'].sum().sort_values(ascending=False).head(5).reset_index()
    build_table(top_items.values.tolist(), ["Product Name", "Total Sales"])

    # Table 2: Sales by Category
    cat_summary = filtered_df.groupby('category')['sales_qntty'].sum().reset_index()
    build_table(cat_summary.values.tolist(), ["Category", "Total Sales"])

    # Table 3: Daily Sales Summary
    daily_summary = filtered_df.groupby('date')['sales_qntty'].sum().reset_index().sort_values(by='date').tail(7)
    build_table(daily_summary.values.tolist(), ["Date", "Total Sales"])

    # Table 4: Forecast Summary (future only)
    if forecast_df is not None and not forecast_df.empty:
        forecast_only = forecast_df[forecast_df['type'] == 'Future Sales'][['date', 'predicted_sales']].head(7)
        build_table(forecast_only.values.tolist(), ["Date", "Predicted Sales"])

    doc.build(elements)
    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), filename="Sales_Summary.pdf")


# Run app
if __name__ == "__main__":
    app.run(debug=True)