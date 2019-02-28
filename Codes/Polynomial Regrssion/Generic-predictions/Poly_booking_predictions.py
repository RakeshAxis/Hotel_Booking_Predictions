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
import matplotlib.pyplot as plot 
import datetime

#comment
def initialize(df):
    df['Jan'] = 0 
    df['Feb'] = 0
    df['Mar'] = 0
    df['Apr'] = 0
    df['May'] = 0
    df['Jun'] = 0
    df['Jul'] = 0
    df['Aug'] = 0
    df['Sep'] = 0
    df['Oct'] = 0
    df['Nov'] = 0
    df['Dec'] = 0
    return df

def format_training_dates(df):
    df = initialize(df)
    for i in df.index:
        stay_dt = df.get_value(i,'stay_dt')
        mm = stay_dt.month
        dd = stay_dt.day
        df.set_value(i,'day',dd)
        week_day = stay_dt.weekday()
        df.set_value(i,'week_day',week_day)
        if(week_day==6 or week_day ==5 or week_day==4):
            df.set_value(i,'weekend',1)
        else:
            df.set_value(i,'weekend',0)
        if mm == 1 : df.set_value(i,'Jan',1)
        elif mm == 2 : df.set_value(i,'Feb',1)
        elif mm == 3 : df.set_value(i,'Mar',1)
        elif mm == 4 : df.set_value(i,'Apr',1)
        elif mm == 5 : df.set_value(i,'May',1)
        elif mm == 6 : df.set_value(i,'Jun',1)
        elif mm == 7 : df.set_value(i,'Jul',1)
        elif mm == 8 : df.set_value(i,'Aug',1)
        elif mm == 9 : df.set_value(i,'Sep',1)
        elif mm == 10 : df.set_value(i,'Oct',1)
        elif mm == 11 : df.set_value(i,'Nov',1)
        elif mm == 12 : df.set_value(i,'Dec',1)
    df = df[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','week_day','day','weekend','rooms']]
    df = df.groupby(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','week_day','day','weekend'],as_index=False)[['rooms']].mean()
    return df

def format_prediction_dates(df):
    df = initialize(df)
    for i in df.index:
        stay_dt = df.get_value(i,'stay_dt')
        mm = stay_dt.month
        dd = stay_dt.day
        df.set_value(i,'day',dd)
        week_day = stay_dt.weekday()
        df.set_value(i,'week_day',week_day)
        if(week_day==6 or week_day ==5 or week_day==4):
            df.set_value(i,'weekend',1)
        else:
            df.set_value(i,'weekend',0)
        if mm == 1 : df.set_value(i,'Jan',1)
        elif mm == 2 : df.set_value(i,'Feb',1)
        elif mm == 3 : df.set_value(i,'Mar',1)
        elif mm == 4 : df.set_value(i,'Apr',1)
        elif mm == 5 : df.set_value(i,'May',1)
        elif mm == 6 : df.set_value(i,'Jun',1)
        elif mm == 7 : df.set_value(i,'Jul',1)
        elif mm == 8 : df.set_value(i,'Aug',1)
        elif mm == 9 : df.set_value(i,'Sep',1)
        elif mm == 10 : df.set_value(i,'Oct',1)
        elif mm == 11 : df.set_value(i,'Nov',1)
        elif mm == 12 : df.set_value(i,'Dec',1)
    df = df[['stay_dt','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','week_day','day','weekend']]
    return df
    

def generate_stay_dt(month,year):
    start_date = pd.to_datetime(str(year)+'-'+str(month)+'-01')
    mm = start_date.month
    df = []
    df = pd.DataFrame(df)
    while (mm == start_date.month):
        df = df.append({'stay_dt':start_date},ignore_index = True)
        start_date+= datetime.timedelta(days=1)
    return df

def yearly_change_factor(df,month,prev_yr):
    
    for i in df.index:
        stay_date = df.get_value(i,'stay_dt')
        mm = stay_date.month
        yy = stay_date.year
        df.set_value(i,'month',mm)
        df.set_value(i,'year',yy)
    df = df.groupby(['year','month'],as_index = False)[['rooms']].sum()
    df = df[(df.month == month) & (df.year <= prev_yr)]
    idx = df.index
    l = idx.size
    if (l <= 1):
        return 0
    j = idx[l-1]
    k = idx[l-2]
    
    factor = (df.get_value(j,'rooms'))/(df.get_value(k,'rooms'))
    if((factor > 1.5) and (factor < 2)):
         factor = factor * (1/22 + 1/19 + 1/18  + 1/17 )
    elif (factor > 2):
         factor = factor*( 1/44 + 1/38 + 1/36 + 1/34)
     
    increment = factor * (df.get_value(j,'rooms'))
    return increment

class Booking_prediction:
    
    # # Wherever booking data is stored :) 
    df_bookings = pd.read_csv("C:\\Users\\Rakesh\\OneDrive\\Documents\\Data\\booking.csv")
    
    ## Constructor takes one argument hotel_id
    def __init__ (self,hotel_id):
        
        # Selecting only the relevant data
        self.hotel_id = hotel_id
        self.booking_data = Booking_prediction.df_bookings[Booking_prediction.df_bookings.hotel_id == hotel_id]
        self.booking_data = self.booking_data[self.booking_data.cancelled == False]
        self.booking_data = self.booking_data.groupby(['stay_dt'],as_index=False)[['rooms']].sum()
        self.booking_data['stay_dt'] = pd.to_datetime(self.booking_data['stay_dt'])
        self.df_train  = self.booking_data.loc[:,:].copy()
        self.df_train = format_training_dates(self.df_train.loc[:,:].copy())
        
        # training the model on the relevant data_set
        x_train = self.df_train.iloc[:,0:15].values
        y_train = self.df_train.iloc[:,15].values
        self.onehot = OneHotEncoder(categorical_features = [12])
        x_train = self.onehot.fit_transform(x_train).toarray()
        
        self.poly = PolynomialFeatures(degree =2)
        x_train_poly = self.poly.fit_transform(x_train)
        self.lin = LinearRegression()
        self.lin.fit(x_train_poly,y_train)
    
    # Month wise predictions only upto nxt year is possible
    # if current data upto 2018 , prediction of 2019 data only , predicting 2020 will give vague results
    def predict(self,month,year):
        pred_df = generate_stay_dt(month,year)
        pred_df = format_prediction_dates(pred_df.loc[:,:].copy())
        pred_x = pred_df.iloc[:,1:16].values
        pred_x = self.onehot.transform(pred_x).toarray()
        pred_x_poly = self.poly.transform(pred_x)
        pred_y = self.lin.predict(pred_x_poly)
        pred_df['prediction'] = pred_y
        
        # Factors in the yearly change in booking numbers
        pred_df = self.manage_increment(pred_df,month,year)
        
        return pred_df
    
     # To fator in change in booking numbers every year 
    def manage_increment(self,df,month,year):
        increment = yearly_change_factor(self.booking_data.iloc[:,:].copy(),month,year-1)
        weekday_increment = 3 * increment / 28 
        weekend_increment = 4 * increment / 28
        for i in df.index:
            prediction = df.get_value(i,'prediction')
            weekend = df.get_value(i,'weekend')
            if(prediction < 0):
                prediction = prediction *(-1)
                df.set_value(i,'prediction',prediction)
            if(weekend ==1):
                df.set_value(i,'prediction',prediction + weekend_increment)
            else:
                df.set_value(i,'prediction',prediction + weekday_increment)
        
        # To smoothen the transition from one day to another
        df = self.normalize(df)
        return df
    
    def normalize(self,df):
        df['avg_prediction'] = 0
        idx= df.index
        first = (df.get_value(idx[0],'prediction')+df.get_value(idx[1],'prediction'))/2
        df.set_value(idx[0],'avg_prediction',first)
        pre= 0 
        cur = df.get_value(idx[0],'prediction')
        nxt = df.get_value(idx[1],'prediction')
        avg = 0
        for i in range(1,idx.size-1):
            pre=cur
            cur=nxt
            nxt=df.get_value(idx[i+1],'prediction')
            avg = (pre+cur+nxt)/3
            df.set_value(idx[i],'avg_prediction',avg)
        
        last = (df.get_value(idx[idx.size-1],'prediction')+df.get_value(idx[idx.size-2],'prediction'))/2
        df.set_value(idx[idx.size-1],'avg_prediction',last)
        return df

# hotel_name = Booking_prediction('hotel_id)
#### To generate booking predictions
# hotel_name_jan19_forecast = hotel_name.predict('01','2019')
# hotel_name_dec19_forecast = hotel_name.predict('02','2019')

hotel_casino = Booking_prediction('57b66e62916bb9001839f1d5')

hotel_casino_dec19_forecast = hotel_casino.predict(12,2018)

hotel_casino_dec19_forecast = hotel_casino_dec19_forecast[['stay_dt','avg_prediction']]

hotel_casino_dec19_forecast.to_csv(index = False)
