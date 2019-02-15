# Booking prediction Using Polynomial Regression

- > Takes 'hotel_id' as argument 

# Formatting of dates 

- > The dates for which prediction is required , are broken down into more columns :
     
     - weekday : values from (0-6) indicating Monday to Sunday
     - weekend : values (0,1)  indicating weekday or weekend
     - month  : 12 more columns are added indicating the month.

# Yearly Change Factor

  - Factors in the increase/decrease in booking trend for the past two years
  - This factor is used to adjust the predicted booking

# Normalization

- > Smoothens out the predicted values using previous and next days booking predictions.
    
      - This is done as there would never be a one day flooding or one day drought of bookings.
