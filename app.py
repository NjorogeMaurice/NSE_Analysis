import datetime
import time
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error , mean_absolute_error, r2_score
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(
    page_title="Stock Data Analysis",
    layout="wide"
)

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def plot_data(stock):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock.index, y=stock['Close'], mode='lines', name='Actual Prices'))
    fig.update_layout(title='Historical Prices over Time',
                      xaxis_title='Date',
                      yaxis_title='Stock Price')
    st.plotly_chart(fig)

def prepare_data(stock_data):
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Date_ordinal'] = stock_data['Date'].map(pd.Timestamp.toordinal)

    X = stock_data[['Date_ordinal','Open','High','Low','Volume','Adj Close']]
    y = stock_data['Close']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train,model):
    model.fit(X_train, y_train)
    return model

# Make predictions
def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def plot_results(X_train, y_train, X_test, y_test, y_pred):
    X_train_dates = X_train[0]
    X_test_dates = X_test[0]
    X_train_dates = X_train[0]
    X_test_dates = X_test[0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train_dates, y=y_train, mode='markers', name='Train Data'))
    fig.add_trace(go.Scatter(x=X_test_dates, y=y_test, mode='markers', name='Test Data'))
    fig.update_layout(title='Distribution of train and test sets',
                      xaxis_title='Date',
                      yaxis_title='Stock Price')
    st.plotly_chart(fig)

# Plot model performance using Plotly
def plot_performance(y_test, y_pred):
    test_dates = y_test.index
    pred_dates = test_dates[:len(y_pred)]  # Get corresponding dates for predicted values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test,mode="markers", name='Actual Prices'))
    fig.add_trace(go.Scatter(x=pred_dates, y=y_pred,mode="markers", name='Predicted Prices'))
    fig.update_layout(title='Actual vs Predicted Prices Over Time',
                      xaxis_title='Date',
                      yaxis_title='Stock Price')
    st.plotly_chart(fig)

def process_data(start,end,option,model_option):
    company = companies[option]
    model = models[model_option]
    stock_data = fetch_stock_data(company,start,end)
    st.divider()
    st.write("##### Top-N rows")
    st.write(stock_data.head())
    plot_data(stock_data)
    st.divider()
    X, y = prepare_data(stock_data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train,model)
    y_pred = make_predictions(model, X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    plot_results(X_train, y_train, X_test, y_test, y_pred)
    st.divider()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plot_performance(y_test, y_pred)
    st.divider()
    st.write("### Model Performance Metrics")
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.write(f'Mean Squared Error (MSE):\n {mse}')
    with col2:
        with st.container(border=True):
            st.write(f'Root Mean Squared Error (RMSE):\n {rmse}')
    with col3:
        with st.container(border=True):
            st.write(f'Mean Absolute Error (MAE):\n {mae}')
    with col4:
        with st.container(border=True):
            st.write(f'R-squared (R2):\n {r2}')

## Main Page
st.header("Stock Price Analysis",divider="blue")
st.write("### Introduction")
st.write("Stock price prediction is a crucial aspect of financial analysis and investment strategy.\
          By accurately forecasting stock prices, investors can make informed decisions, minimize risks, and maximize returns. \
         We utilize various machine learning models to perform the predictions over a specified time and visualize the results using interactive plots.")
st.divider()

companies = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon.com, Inc.": "AMZN",
    "Alphabet Inc. (Google)": "GOOGL",
    "Tesla, Inc.": "TSLA",
    "Meta Platforms, Inc. (Facebook)": "META",
    "NVIDIA Corporation": "NVDA",
    "Berkshire Hathaway Inc.": "BRK-B",
    "Johnson & Johnson": "JNJ",
    "Procter & Gamble Co.": "PG"
}

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
}

option = st.selectbox(
"Select Company to do stock analysis",
(companies.keys()))
model_option = st.selectbox(
"Select baseline model",
(models.keys()))
start = st.date_input("Select start date", datetime.date(2019, 7, 6))
end = st.date_input("Select end date", datetime.date(2019, 7, 6))
if end > datetime.datetime.now().date() or start > datetime.datetime.now().date():
    st.warning("Start or end dates cannot be later than today")
elif start > end:
    st.warning("Start date cannot be later than end date")
elif start == end:
    st.warning("Start and end dates cannot be the same")
else:
    if st.button("Proceed"):
        placeholder = st.empty()
        placeholder.write("Processing....")
        process_data(start,end,option,model_option)
        placeholder.text("Complete")

st.divider()

## Sidebar
add_sidebar_title = st.sidebar.write("## Stock Price Analysis")
add_divider = st.sidebar.divider() 



