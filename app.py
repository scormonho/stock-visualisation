import streamlit as st 
import yfinance as yf
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from cProfile import label


st.set_page_config(
    page_title="Hack4Pan",
    page_icon="https://cdn-icons-png.flaticon.com/128/3094/3094918.png",
    initial_sidebar_state='collapsed'
)

def get_user_input():
    input = st.text_input('Enter a stock', 'TSLA')
    return input

@st.cache(allow_output_mutation=True)
def fetch_chart_data(input):
    data = yf.download(input)
    dataDF = yf.Ticker(input)
    return data, dataDF

def display_chart(data_close):
    st.subheader('Interactive chart - Closing Price')
    st.line_chart(data_close)
    
def predict_stock(data, date):
    predictStock = data.drop(["Open", "High", "Low", "Adj Close"], axis = 1)
    new_dataset = predictStock.drop(["Volume"], axis = 1, inplace = False)
    predictDate = str(date)
    
    # Use date to select training set
    target = new_dataset[(new_dataset.index <= predictDate)]
    train_data = target.values
    valid_data = new_dataset[(new_dataset.index > predictDate)]
    
    # Feature scaling on training set
    scaler = MinMaxScaler(feature_range=(0,1))
    dataScaler = scaler.fit(train_data)
    train_data = dataScaler.transform(train_data)


    x_train_data,y_train_data = [],[]
    for i in range(60,len(train_data)):
        x_train_data.append(train_data[i-60:i,0])
        y_train_data.append(train_data[i,0])

    # convert training sets to numpy arrays
    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    # reshape x_train_data vertically
    x_train_data = np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.fit_transform(inputs_data)
    lstm_model.compile(loss='mean_squared_error',optimizer='adam')
    lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)
    
    X_test=[]
    for i in range(60,inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i,0])
    X_test=np.array(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    predicted_closing_price=lstm_model.predict(X_test)
    predicted_closing_price=scaler.inverse_transform(predicted_closing_price)
    
    lstm_model.save("saved_model.h5")
    
    train_data=new_dataset[(new_dataset.index <= predictDate)]
    valid_data=new_dataset[(new_dataset.index > predictDate)]
    valid_data['Predictions']= predicted_closing_price

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    sb.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 
               'axes.labelcolor': 'white', 'axes.edgecolor': 'white',
               'legend.labelcolor':'white','legend.loc':'best'})
    sb.lineplot(data=predictStock['Close'], ax=ax1, alpha=1.0, color='orange', legend="auto", label="Close")
    sb.lineplot(data=valid_data["Predictions"], ax=ax1, alpha=1.0, color="#58FF4B", legend="auto", label="Predicted")
    sb.lineplot(data=predictStock['Volume'], ax=ax2, alpha=0.4, color='yellow', legend="auto", label="Volume")
    plt.show()
    st.subheader("Predicted chart")
    st.pyplot(fig,use_container_width=True)
    st.markdown('---')
    
    
st.image("https://cdn.substack.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4b8647a-09e8-4576-95de-e622dcc38d72_1280x720.jpeg")
st.title("Welcome to the Stock App!")
with st.expander("How To Use"):
    st.markdown("First, type in the **ticker** symbol you're looking for.")  
    st.markdown("Next, click the arrow icon on the top left corner. This opens up a sidebar.")             
    st.markdown("Select the date you wish to start predicting the stock price from.")
    st.markdown("Give a moment for the model to process your input.")              
    st.markdown("And that's it! The prediction will appear magically before your eyes. It's that **easy**!")            

# get user input (in sidebar)
st.header("Which stock would you like to see today?")
stock = get_user_input()

# fetch chart data
chart, chartDF = fetch_chart_data(stock)
chart_close = chart.drop(['High','Low','Adj Close','Open','Volume'], axis = 1)

# display chart & info
st.header(stock)
string_summary = chartDF.info['longBusinessSummary']
st.info(string_summary)
st.markdown("---")
display_chart(chart_close)

# prediction date
st.sidebar.subheader('Choose a date to start predicting')
min_days = chart.index[0] + datetime.timedelta(days=60)
default = chart.index[-1] - datetime.timedelta(days=1)
prediction_date = st.sidebar.date_input("Start date", value=default, min_value=min_days, max_value=chart.index[-1])
dt64 = np.datetime64(prediction_date)
dt64_converted = pd.Timestamp(dt64)
st.sidebar.write("Selected date: ", dt64)

# fetch plot data
plotData = chart.drop(["Open", "High", "Low", "Adj Close"], axis = 1)

# plot static chart 
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

sb.set(rc={'axes.facecolor':'#0E1117', 'figure.facecolor':'#0E1117',
           'axes.labelcolor':'white', 'xtick.color':'white','ytick.color':'white',
           'axes.labelweight':'bold','legend.labelcolor':'white','legend.loc':'best'})
sb.lineplot(data=plotData['Close'], ax=ax1, alpha=1.0, color='orange', legend="auto", label="Close")
sb.lineplot(x=plotData.index, y=plotData['Volume'], data=plotData['Volume'], color='yellow', 
            ax=ax2, alpha=0.6, legend="auto", label="Volume")
plt.show()

st.subheader("Static chart - Closing Price & Volume")
st.pyplot(fig,use_container_width=True)
st.markdown("---")
############################

# machine learning to predict
if st.sidebar.button("Predict"):
    with st.spinner("Please wait..."):
        predict_stock(chart, dt64_converted)
