
# coding: utf-8

# In[1]:

# hotel_id = 57b66e62916bb9001839f1d5 , of casino hotel
# Polynomial Regression model

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import calendar
import pymongo
import urllib.parse
from datetime import date
from pandas import DatetimeIndex
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import matplotlib.pyplot as plt


# In[2]:

def get_booking_data(hotel_id):
    booking_data = df_bookings[df_bookings.hotel_id == hotel_id]
    booking_data = booking_data[booking_data.cancelled == False]
    booking_data = booking_data.groupby(['stay_dt'],as_index=False)[['rooms']].sum()
    booking_data['stay_dt'] = pd.to_datetime(booking_data['stay_dt'])
    return booking_data


# In[3]:

df_bookings = pd.read_csv("C:\\Users\\Rakesh\\OneDrive\\Documents\\Data\\booking.csv")
#df_hotels = pd.read_csv("C:\\Users\\Rakesh\\OneDrive\\Documents\\Data\\hotel.csv")

# Testing for casino hotel

hotel_id = '57b66e62916bb9001839f1d5'
booking_data = get_booking_data(hotel_id)
#hotel_data = get_hotel_data(hotel_id)


# In[59]:

def prev_yr_month(df):
    a = []
    for i in df.index:
        yy = df.get_value(i,'year')
        mm = df.get_value(i,'month')
        prev_yr = df.loc[(df['year']==yy-1)&(df['month']==mm)]['rooms']
        if(prev_yr.size > 0):
            df.set_value(i,'prev_yr_bookings',prev_yr)
        else:
            a= np.append(a,i)
    df = df.drop(a)
    return df


# In[60]:

def monthwise(df):
    for i in df.index:
        stay_dt = df.get_value(i,'stay_dt')
        mm = stay_dt.month
        yy = stay_dt.year
        df.set_value(i,'month',mm)
        df.set_value(i,'year',yy)
    df = df.groupby(['month','year'],as_index=False)[['rooms']].sum()
    df['prev_yr_bookings'] = 0.0
    df = prev_yr_month(df)
    df = df[['year','month','prev_yr_bookings','rooms']]
    df = df.sort_values(by=['year', 'month'],ascending =True)
    return df


# In[61]:

df_booking = monthwise(booking_data.copy())

df_train = df_booking[df_booking.year<2018]

df_test = df_booking[(df_booking.year>=2018) & (df_booking.year<2019)]


# In[62]:

df_train.head()


# In[63]:

train_x = df_train.iloc[:,1:3].values
train_y = df_train.iloc[:,3].values
test_x = df_test.iloc[:,1:3].values
test_y = df_test.iloc[:,3].values


# In[73]:

month_encoder = OneHotEncoder(categorical_features = [0])

train_x = month_encoder.fit_transform(train_x).toarray()
test_x = month_encoder.transform(test_x).toarray()


# In[74]:

train_x.shape


# In[75]:

poly = PolynomialFeatures( degree = 1 )
x_train_poly = poly.fit_transform(train_x)
lin = LinearRegression()
lin.fit(x_train_poly , train_y)

x_test_poly = poly.transform(test_x)
y_pred = lin.predict(x_test_poly)

test_pred_df = df_test


# In[76]:

test_pred_df['predict'] = y_pred


# In[77]:

test_pred_df


# In[78]:

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(test_pred_df['rooms'], test_pred_df['predict'])))


# In[ ]:



