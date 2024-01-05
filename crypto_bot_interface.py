import pandas as pd
import numpy as np 
import math
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import base64
import pydeck as pdk
import altair as alt
import base64

import cryptocompare

from itertools import cycle
#from prophet import Prophet
from pandas import read_csv
from sklearn.linear_model import Ridge
import pickle
import joblib

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sb
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from prophet.plot import plot_plotly, plot_components_plotly

from datetime import date
import datetime
import numpy as np
import pandas as pd
#import plotly.graph_objects module
# Import datetime package
from datetime import datetime

# Import matplotlib and set the style for plotting
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('seaborn-darkgrid')
sns.set(rc={'figure.figsize':(25,8)})
fig, ax = plt.subplots(figsize=(25, 8))
st.cache(allow_output_mutation=True)


st.markdown(f'<h1 style="color:black;font-size:50px;">Projet Crypto Bot promo 2 Data Engineer</h1>', unsafe_allow_html=True)


# Get the API key from the Quantra file located inside the data_modules folder
cryptocompare_API_key = '1e8d05fbc87cd090fa5c22977e08fa4cab57dc2d799fef08d8a1c5ebcfaa1f97'

# Set the API key in the cryptocompare object
cryptocompare.cryptocompare._set_api_key_parameter(cryptocompare_API_key)

#print("API Key set!")


# Fetch the raw ticker list
raw_ticker_data = cryptocompare.get_coin_list()

# Convert the raw data from dictionary format to DataFrame
all_tickers = pd.DataFrame.from_dict(raw_ticker_data).T

# Preview the first 6 columns and the last 5 rows of the ticker list
#all_tickers.iloc[:, :5]

#st.sidebar.markdown(, unsafe_allow_html=True)

options=all_tickers['Name'].to_list() #8708

st.sidebar.markdown(f'<h1 style="color:black;font-size:20px;">Choisir la monnaie</h1>', unsafe_allow_html=True)
option = st.sidebar.selectbox('', options)

ticker_symbol = option #'BTC'  #'ADA'
currency = 'USD'
exchange_name = 'CCCAGG'

#st.write('Vous avez choisit:', option)
from datetime import date

today = date.today()

# dd/mm/YY
d = today.strftime("%d")
m = today.strftime("%m")
an = today.strftime("%Y")

#print("d1 =", int(d))
#print("d1 =", int(m))
#print("d1 =", int(an))

##### CHOIX DE LA MONNAIE 



# HISTORIQUE DES DONNEES DEPUIS 2015


today = date.today()

d = 1#today.strftime("%d")
m = 1#today.strftime("%m")
an = 2016#today.strftime("%Y")

# Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = 365
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)

# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data_2015 = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data_2015.set_index('time', inplace=True)
hourly_price_data_2015.index = pd.to_datetime(hourly_price_data_2015.index, unit='s')
hourly_price_data_2015['datetimes'] = hourly_price_data_2015.index
hourly_price_data_2015['datetimes'] = hourly_price_data_2015['datetimes'].dt.strftime('%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data_2015=hourly_price_data_2015.reset_index()#.iloc[:, :6].last()

#HISTORIQUE ANNEE 2016

today = date.today()

d = 1#today.strftime("%d")
m = 1#today.strftime("%m")
an = 2017#today.strftime("%Y")

# Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = 365
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)

# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data_2016 = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data_2016.set_index("time", inplace=True)
hourly_price_data_2016.index = pd.to_datetime(hourly_price_data_2016.index, unit='s')
hourly_price_data_2016['datetimes'] = hourly_price_data_2016.index
hourly_price_data_2016['datetimes'] = hourly_price_data_2016['datetimes'].dt.strftime(
    '%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data_2016=hourly_price_data_2016.reset_index()#.iloc[:, :6].last()

#HISTORIQUE ANNEE 2017

today = date.today()

d = 1#today.strftime("%d")
m = 1#today.strftime("%m")
an = 2018#today.strftime("%Y")

# Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = 365
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)

# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data_2017 = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data_2017.set_index("time", inplace=True)
hourly_price_data_2017.index = pd.to_datetime(hourly_price_data_2017.index, unit='s')
hourly_price_data_2017['datetimes'] = hourly_price_data_2017.index
hourly_price_data_2017['datetimes'] = hourly_price_data_2017['datetimes'].dt.strftime(
    '%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data_2017=hourly_price_data_2017.reset_index()#.iloc[:, :6].last()


#HISTORIQUE ANNEE 2018

today = date.today()

d = 1#today.strftime("%d")
m = 1#today.strftime("%m")
an = 2019#today.strftime("%Y")

# Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = 365
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)

# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data_2018 = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data_2018.set_index("time", inplace=True)
hourly_price_data_2018.index = pd.to_datetime(hourly_price_data_2018.index, unit='s')
hourly_price_data_2018['datetimes'] = hourly_price_data_2018.index
hourly_price_data_2018['datetimes'] = hourly_price_data_2018['datetimes'].dt.strftime(
    '%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data_2018=hourly_price_data_2018.reset_index()#.iloc[:, :6].last()

#HISTORIQUE ANNEE 2019

today = date.today()

d = 1#today.strftime("%d")
m = 1#today.strftime("%m")
an = 2020#today.strftime("%Y")

# Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = 365
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)

# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data_2019 = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data_2019.set_index("time", inplace=True)
hourly_price_data_2019.index = pd.to_datetime(hourly_price_data_2019.index, unit='s')
hourly_price_data_2019['datetimes'] = hourly_price_data_2019.index
hourly_price_data_2019['datetimes'] = hourly_price_data_2019['datetimes'].dt.strftime(
    '%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data_2019=hourly_price_data_2019.reset_index()#.iloc[:, :6].last()

#HISTORIQUE ANNEE 2020

today = date.today()

d = 1#today.strftime("%d")
m = 1#today.strftime("%m")
an = 2021#today.strftime("%Y")

# Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = 365
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)

# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data_2020 = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data_2020.set_index("time", inplace=True)
hourly_price_data_2020.index = pd.to_datetime(hourly_price_data_2020.index, unit='s')
hourly_price_data_2020['datetimes'] = hourly_price_data_2020.index
hourly_price_data_2020['datetimes'] = hourly_price_data_2020['datetimes'].dt.strftime(
    '%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data_2020=hourly_price_data_2020.reset_index()#.iloc[:, :6].last()


#HISTORIQUE ANNEE 2021
today = date.today()

d = 1#today.strftime("%d")
m = 1#today.strftime("%m")
an = 2022#today.strftime("%Y")

# Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = 365
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)

# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data_2021 = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data_2021.set_index("time", inplace=True)
hourly_price_data_2021.index = pd.to_datetime(hourly_price_data_2021.index, unit='s')
hourly_price_data_2021['datetimes'] = hourly_price_data_2021.index
hourly_price_data_2021['datetimes'] = hourly_price_data_2021['datetimes'].dt.strftime(
    '%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data_2021=hourly_price_data_2021.reset_index()#.iloc[:, :6].last()



#HISTORIQUE ANNEE 2022
from datetime import date
today = date.today()

d = 1#today.strftime("%d")
m = 1#today.strftime("%m")
an = 2023#today.strftime("%Y")

# Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = 365
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)

# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data_2022 = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data_2022.set_index("time", inplace=True)
hourly_price_data_2022.index = pd.to_datetime(hourly_price_data_2022.index, unit='s')
hourly_price_data_2022['datetimes'] = hourly_price_data_2022.index
hourly_price_data_2022['datetimes'] = hourly_price_data_2022['datetimes'].dt.strftime(
    '%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data_2022=hourly_price_data_2022.reset_index()#.iloc[:, :6].last()

#DONNEES STREAMING

#extraction nbr de Dates dynamique pas vraiment du streaming :) il fait l'affaire

an_22, m_22, d_22=2023,1,1
d0 = date(an_22, m_22, d_22)

d = int(today.strftime("%d"))
m = int(today.strftime("%m"))
an = int(today.strftime("%Y"))

d1 = date(an, m, d)
delta = d1 - d0
#print(delta.days)


today = date.today()

# dd/mm/YY
from datetime import date

d0 = date(an_22, m_22, d_22)

d = int(today.strftime("%d"))
m = int(today.strftime("%m"))
an = int(today.strftime("%Y"))

d1 = date(an, m, d)
delta = d1 - d0
#print("d1 =", int(d))
#print("d1 =", int(m))
#print("d1 =", int(an))

#Define the ticker symbol and other details
#ticker_symbol = 'BTC'  #'ADA'
#currency = 'USD'
limit_value = delta.days
#exchange_name = 'CCCAGG'
data_before_timestamp = datetime(int(an),int(m), int(d), 0, 0)


# Fetch the raw price data
raw_price_data = \
    cryptocompare.get_historical_price_day(
        ticker_symbol,
        currency,
        limit=limit_value,
        exchange=exchange_name,
        toTs=data_before_timestamp
    )

# Convert the raw price data into a DataFrame
hourly_price_data = pd.DataFrame.from_dict(raw_price_data)

# Set the time columns as index and convert it to datetime
hourly_price_data.set_index("time", inplace=True)
hourly_price_data.index = pd.to_datetime(hourly_price_data.index, unit='s')
hourly_price_data['datetimes'] = hourly_price_data.index
hourly_price_data['datetimes'] = hourly_price_data['datetimes'].dt.strftime(
    '%Y-%m-%d')

# Preview the last 5 values of the the first 7 columns of the DataFrame
hourly_price_data=hourly_price_data.reset_index()#.iloc[:, :6].last()


######### concatenation des données historique et streaming


#list dataframe to append
frame = [hourly_price_data_2016, hourly_price_data_2017, hourly_price_data_2018, hourly_price_data_2019,hourly_price_data_2020,hourly_price_data_2021,hourly_price_data_2022]

#new dataframe to store append result
hist_data=pd.concat(frame, axis=0)

hist_data=pd.DataFrame(hist_data).reset_index().drop_duplicates(hist_data.columns)
hourly_price_data=pd.DataFrame(hourly_price_data).reset_index().drop_duplicates(hourly_price_data.columns)

crypto_data=pd.concat([hist_data,hourly_price_data],axis=0)

del crypto_data ['conversionType']
del crypto_data ['conversionSymbol']
del crypto_data ['datetimes']
del crypto_data ['volumefrom']
del crypto_data ['index']
#del crypto_data ['time']
#crypto_data=crypto_data[['times','open','low','high','close','volumeto']]
#crypto_data=crypto_data.reset_index()
crypto_data['time'] = crypto_data['time'].dt.strftime('%Y-%m-%d')
st.subheader("Graphe de suivi de la crypto monnaie")

fig = plt.subplots(figsize=(30, 7))
fig = go.Figure(data=[go.Candlestick(x=crypto_data['time'],
    close=crypto_data.close,
    high=crypto_data.high,
    low=crypto_data.low,
    open=crypto_data.open)])
fig.update_layout(title = {
    'text': 'Graphique prix de la crypto monnaie en $',
    'y':0.90,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'})
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Prix ($)')
fig.update_layout(xaxis_rangeslider_visible=True,width=500,
    height=500
) # Set Set Range Slider Bar and Title
st.plotly_chart(fig,use_container_width=True)#.show()

st.write(crypto_data)

###########  download data  'Download Started!'
crypto_data_csv = crypto_data.to_csv(index=False)
crypto_data_b64 = base64.b64encode(crypto_data_csv.encode()).decode()  # some strings
linko= f'<a href="data:file/csv;base64,{crypto_data_b64}" download="crypto_data.csv">Télecharger le dataset en format csv</a>'
st.markdown(linko, unsafe_allow_html=True)

#################EXPLORATION DES DONNEES
st.subheader("Exploration des données")
# Types de données

#st.markdown("Types de données")
#info=crypto_data.info()

#st.write(info)# description des données numériques

st.markdown("Description des données numériques")
describe=pd.DataFrame(crypto_data.describe())
st.write(describe)
# Cellules vides
#st.write(f"Le nombre de cellule vides dans notre dataset est :\n{crypto_data.isna().sum()}")

st.markdown('Les données n ont pas de valeurs manquantes, nous n effectuerons donc pas un nettoyage des données.')


############# TRANSFORMATION DE DONNEES

st.subheader("Transformation des données")
#Nous définissons la colonne <b>Date</b> comme index de notre Dataframe

crypto_data = crypto_data.set_index('time', drop=True)


# VISUALISATION DE LA VALEUR "VOLUME"
# visualiser la valeur "Volume"

st.markdown("Visualisation de la valeur du Volume")


#btc_trace = go.Figure(data=[go.Scatter(x=crypto_data.index, y=crypto_data['volumeto'], name= 'Volume')])
#btc_trace = plt.subplots(figsize=(30, 7))
#btc_trace.update_layout(
#title = {
#    'text': 'Volume de la crypto monnaie',
#    'y':0.90,
#    'x':0.5,
#    'xanchor': 'center',
#    'yanchor': 'top'} , width=500,
#    height=500,
#template="plotly_white")
#st.plotly_chart(btc_trace,use_container_width=True)

def vol_traded(data ,title,color):
    area = px.area(data_frame=data,
               x = data.index ,
               y = "volumeto",
               markers = True)
    area.update_traces(line_color=color)
    area.update_xaxes(
        title_text = 'Date',
        rangeslider_visible = True)
    area.update_yaxes(title_text = 'Volume de la crypto monnaie')
    area.update_layout(showlegend = True, xaxis_rangeslider_visible=False,
        title = {
            'text': '{:} Volume Trader'.format(title),
            'y':0.94,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        template="plotly_white")
    return area
#vol_traded(crypto_data[-delta.days:], "Bitcoin",color = "blue")
st.plotly_chart(vol_traded(crypto_data[-delta.days:], "Crypto monnaie",color = "blue"),use_container_width=True)
#Traçons la matrice de corrélation pour sélectionner les colonnes appropriées.

# the correlation matrix 
mat = crypto_data.corr()
plt.figure(figsize=(17, 13))
sb.heatmap(mat,annot=True)

#plt.show()
# pour la partie avec prophet et garder la colonne time avec le meme format du début
prophet_Data=crypto_data.copy()
###################  APPROCHE AVEC DU MACHINE LEARNING
crypto_data=crypto_data.reset_index()
crypto_data['time'] = pd.to_datetime(crypto_data['time'])
crypto_data['time'] = (crypto_data['time'].astype(np.int64) / 1e9).astype(float)
#crypto_data['time'] = (crypto_data['time'].astype(np.int64) / 1e9).astype(float)
#crypto_data['time']=crypto_data['time'].dt.total_seconds()
#st.write(crypto_data)
BTC_Data=crypto_data.copy()

BTC_Data=BTC_Data.reset_index()

#Dataframe du prix de close du Bitcoin.
closedf = BTC_Data[['time','close']]
#st.write("Forme du dataframe close", closedf.shape)

#validation_set = closedf[closedf['time'] >= '2022-01-01']

#closedf = closedf[closedf['time'] > '2014-12-31']
close_stock = closedf.copy()
#st.write("Données totales pour la prédiction: ",closedf.shape[0])
del closedf['time']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
#print(closedf.shape)
#Maintenant on va diviser notre jeu de données 70% pour l'entraînement et 30% pour le test

training_size=int(len(closedf)*0.70)
test_size=len(closedf)-training_size
train_data=closedf[0:int(training_size),:]
test_data=closedf[int(training_size):len(closedf),:1]

st.write("la taille des données d'entraînement est: ",train_data.shape[0])
st.write("la taille d'échantillon du test est: ",test_data.shape[0])

##############  VISUALISATION TRAIN ET TEST DATA



#close_stock['time']=pd.to_datetime(close_stock['time'], unit='s')
#sb.lineplot(x = close_stock['time'][:train_data.shape[0]], y = close_stock['close'][:train_data.shape[0]], color = 'black')
#sb.lineplot(x = close_stock['time'][train_data.shape[0]:], y = close_stock['close'][train_data.shape[0]:], color = 'red')

# Formatting
#ax.set_title('Train & Test data', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
#ax.set_xlabel('Date', fontsize = 16, fontdict=dict(weight='bold'))
#ax.set_ylabel('Prix btc', fontsize = 16, fontdict=dict(weight='bold'))
#plt.tick_params(axis='y', which='major', labelsize=16)
#plt.tick_params(axis='x', which='major', labelsize=16)
#plt.legend(loc='upper right' ,labels = ('train', 'test'))
#st.plotly_chart(fig,use_container_width=True)


def candelstick_chart(data,title):
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data['time'] = data['time'].dt.strftime('%Y-%m-%d')
    candlestick = go.Figure(data = [go.Candlestick(x =data[('time')], 
                                               open = data[('open')], 
                                               high = data[('high')], 
                                               low = data[('low')], 
                                               close = data[('close')],
                                               #increasing_line_color= 'cyan', 
                                               #decreasing_line_color= 'gray'
                                                )])
    candlestick.update_xaxes(title_text = 'Date',
                             rangeslider_visible = True)

    candlestick.update_layout(
    title = {
        'text': '{:} Candelstick Chart'.format(title),
        'y':0.90,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'} , 
    template="plotly_white")

    candlestick.update_yaxes(title_text = 'Prix en $', ticksuffix = '$')
    return candlestick

btc_plot = candelstick_chart(crypto_data[-80:],title = option)
st.plotly_chart(btc_plot,use_container_width=True)

#Convertir un tableau de valeurs en une matrice de jeu de données


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   #i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 1
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


#st.write("X_train: ", X_train.shape)
#st.write("y_train: ", y_train.shape)
#st.write("X_test: ", X_test.shape)
#st.write("y_test", y_test.shape)


#########  CONSTRUCTION DU MODELE


LR=LinearRegression()
models = {"Linear Regression":LR}

# Step 3: Train the model
LR.fit(X_train, y_train)

# Step 4: Make predictions
predictions = LR.fit(X_train, y_train).predict(X_test)
st.write("l'entrainement de notre model a donné")
st.write("Mean Absolute Error - MAE : " + str(mean_absolute_error(y_test, predictions)))
st.write("Root Mean squared Error - RMSE : " + str(math.sqrt(mean_squared_error(y_test, predictions)))+"\n")




train_predict=LR.fit(X_train, y_train).predict(X_train)
test_predict=LR.fit(X_train, y_train).predict(X_test)
    # Transform back to original form
train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

close_stock['time']=pd.to_datetime(close_stock['time'], unit='s')
### On va évaluser nos modeles à la fois sur les données d'endtrainement et de test et d'obtenir un score de précision
def eval(model):
    #train_predict, test_predict# = predict(model)
    # shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
    

    
    names = cycle(['Prix close original','Prix predit Train_close','Prix predit Test_close'])
    close_stock['time']=pd.to_datetime(close_stock['time'], unit='s')
    plotdf = pd.DataFrame({'date': close_stock['time'],
                           'close_original': close_stock['close'],
                          'train_close_predit': trainPredictPlot.reshape(1,-1)[0].tolist(),
                          'test_close_predit': testPredictPlot.reshape(1,-1)[0].tolist()})
    #fig = plt.subplots(figsize=(30, 7))
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['close_original'],plotdf['train_close_predit'],
                                              plotdf['test_close_predit']],
                  labels={'value':'Prix close','date': 'Date'})
    fig.update_layout(
        title = {
    'text': 'Prediction avec '+model,
    'y':0.90,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Prix close')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig,use_container_width=True)
    #fig.show()
    
    val=explained_variance_score(original_ytrain, train_predict)*100
    val2=explained_variance_score(original_ytest, test_predict)*100
    #st.write("{:.2f}".format(val)+"%")
    ## Score de la variance pour la regression
    #st.write("\n\nScore de régression de variance expliqué par les données d'entraînement:", "{:.4f}".format(val)+"%")
    #st.write("Score de régression de variance expliqué par les données de test:", "{:.4f}".format(val2)+"%" )

    ## Score R square pour la regression
    val3=r2_score(original_ytrain, train_predict)*100
    val4=r2_score(original_ytest, test_predict)*100
    st.write("\nScore R2 des données d'entraînement:", "{:.4f}".format(val3)+"%")
    st.write("\n Score R2 des données de test:",  "{:.4f}".format(val4)+"%")
    #print(train_predict, test_predict)
    #print(plotdf)
    #data_predict = pd.DataFrame({'time':close_stock['time'],'original_ytrain':original_ytrain,'train_predict':train_predict,
    #            'original_ytest':original_ytest,'test_predict':test_predict})
    #print(df)
for model in models:
    eval(model) 

##########  prédiction future de J+1 avec la libraire Prophet de Facebook

arima_data=crypto_data.copy()#.reset_index()
del arima_data['high']
del arima_data['low']
del arima_data['open']
del arima_data['volumeto']
arima_data=arima_data[['time','close']]


#### VALIDATION SET
#st.markdown("étape de validation")
training_size=int(len(arima_data)*0.80)
test_size=len(arima_data)-training_size

split_point = len(arima_data) - test_size
dataset, validation = arima_data[0:split_point], arima_data[split_point:]
#st.write('Dataset %d, Validation %d' % (len(dataset), len(validation)))
#dataset.to_csv('dataset.csv', header=False)
#validation.to_csv('validation.csv', header=False)

scaler=MinMaxScaler(feature_range=(0,1))
validation=scaler.fit_transform(np.array(validation['close']).reshape(-1,1))
#st.write(validation.shape)
# Sauvegarde du modele en format pickel

# Import modules

# Assuming 'model' is your trained Linear Regression model
joblib.dump(LR.fit(X_train, y_train), 'linear_regression_model.pkl')

# Load the saved model
loaded_model = joblib.load('linear_regression_model.pkl')

# Make predictions on the new data
# Making Predictions 
Y_pred = loaded_model.predict(validation)

# Metric Result
mse = mean_squared_error(validation, Y_pred)
st.write("la RMSE sur le dataset de validation est : " + str(math.sqrt(mse))+"\n")





from prophet import Prophet
prophet_Data=prophet_Data.reset_index()

df = prophet_Data[['time', 'close']]
df.columns = ['ds', 'y']

########################################################

#Python
m = Prophet(changepoint_range=1,changepoint_prior_scale=0.75)
m.fit(df)

#Python

jr = st.slider('Choisir le nombre de jour à prédire', 1, 365,1)
st.write("Vous avez choisit ", jr, 'Jours à prédire')

future = m.make_future_dataframe(periods=jr)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
st.subheader("Graphe de la tendance et la saisonnalité.")
st.write(plot_components_plotly(m,forecast, uncertainty=True))

forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
####################################################
# Python

# Define the end date
end_date = pd.to_datetime(forecast['ds'].max())
#LONGEUR DE LA TABLE STREAMING ANNEE 2023
x=len(hourly_price_data)/7
# Generate a date range with a frequency of 7 days
date_range = pd.date_range(end=end_date, periods=x, freq='7D')

# Print the date range
period=pd.DataFrame(date_range)
period.columns=['ds']
period['ds'] = period['ds'].dt.strftime('%Y-%m-%d')

new_df=pd.merge(df,period,on=['ds'],how='right')
mean_val=new_df['y'].mean()
std_val=new_df['y'].std()
seuil_min=abs(mean_val-std_val)
seuil_max=abs(mean_val+std_val)
# yhat valeur prédite
yhat_val=forecast[forecast['ds']==forecast['ds'].max()]['yhat'].values[0]
###############################

st.subheader("Graphe de la prévision.")

st.markdown(f'<h1 style="font-size:15px;">Prévision sur {jr} jours:</h1>', unsafe_allow_html=True)

st.plotly_chart(plot_plotly(m, forecast))
################################



#st.markdown("le seuil minimal est:",)
st.markdown(f'<h1 style="font-size:18px;">Le seuil minimal est:</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:red;font-size:18px;">{seuil_min}</h1>', unsafe_allow_html=True)

st.markdown(f'<h1 style="font-size:18px;">La valeur prédite est:</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:blue;font-size:18px;">{yhat_val}</h1>', unsafe_allow_html=True)

st.markdown(f'<h1 style="font-size:18px;">Le seuil maximal est:</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:green;font-size:18px;">{seuil_max}</h1>', unsafe_allow_html=True)

if (yhat_val>=seuil_min and yhat_val<seuil_max) or (yhat_val>seuil_max) :
    #txt='''valeur de la monnaie évolue'''
    st.markdown(f'<h1 style="font-size:18px;">Valeur de la monnaie évolue</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:green;font-size:18px;">Vente</h1>', unsafe_allow_html=True)
    #st.write('valeur de la monnaie évolue : vente')
else:
    st.markdown(f'<h1 style="font-size:18px;">Valeur de la monnaie diminue</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:18px;">Achat</h1>', unsafe_allow_html=True)


#seuil_min: 23348.104932414062
#valeur prédite: 28289.009879449568
#seuil_max: 35049.29467542908
#valeur de la monnaie évolue : vente

st.sidebar.markdown('<a href="mohamedamine.bellalij@orange.com">Contact me !</a>', unsafe_allow_html=True)