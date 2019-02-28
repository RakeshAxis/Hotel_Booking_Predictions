# How to use ?

# 1. Create an object of the class Booking_prediction.
    
     Pass the hotel_id when creating the object for a hotel .
      e.g. :  hotel_casino = Booking_prediction('57b66e62916bb9001839f1d5')
    
# 2. How to get predictions. 
    1.> Create a variable that stores the predictions in a DataFrame.
    
    2.> Get the predictions by using the method preidct . 
            
            * Predict argument takes month number and year for which forecasting is required.
         
         e.g.   hotel_casino_dec19_forecast =  hotel_casino.predict(12,2019)
               
               var  hotel_casino_dec19_forecast stores the predictions for December 2019 in a DataFrame.
               
               hotel_casino_dec19_forecast.to_csv( index = False ) , should be used to get the forecasts in a CSV file
