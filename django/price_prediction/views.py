# price_prediction/views.py
from django.shortcuts import render, redirect
import pandas as pd
from num2words import num2words
import os
from django.conf import settings 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Page 1: Select City
def select_city(request):
    if request.method == 'POST':
        city = request.POST['city']
        print("request",request)
        print("city",city)
        request.session['city'] = city  # Store in session
        return redirect('select_details')
    return render(request, 'welcome.html')

# Page 2: Input Details
def select_details(request):
    if request.method == 'POST':
        # Collect form data
        request.session['bhk'] = int(request.POST['bhk'])
        request.session['furnishing'] = request.POST['furnishing']
        request.session['property_type'] = request.POST['property_type']
        return redirect('predict_price')
    return render(request, 'details.html')
    

# Page 3: Show Predicted Price
def predict_price(request):
    city = request.session.get('city')
    bhk = request.session.get('bhk')
    furnishing = request.session.get('furnishing')
    property_type = request.session.get('property_type')

    csv_path = os.path.join(settings.BASE_DIR, 'price_prediction', 'out.csv')
    
    # Load data from CSV
    df = pd.read_csv(csv_path)

    # Load and clean data (assuming 'df' is already loaded)
    df = df.dropna(subset=['Price_Unit', 'Furnishing'])

    # Convert price to a single unit (e.g., Lac)
    def convert_price(row):
        return row['Price_Amount'] * 100 if row['Price_Unit'] == 'Cr' else row['Price_Amount']

    df['Price_Lac'] = df.apply(convert_price, axis=1)

    # Drop unnecessary columns
    df = df.drop(columns=['Price', 'Price_Amount', 'Price_Unit', 'Link', 'Address', 'Title'])

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['City', 'Furnishing', 'Property_Type'], drop_first=True)

    # Prepare target and features
    X = df.drop(columns=['Price_Lac'])
    y = df['Price_Lac']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training with Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Mean Absolute Error (Linear Regression): {mae}")
    print(f"Root Mean Square Error (Linear Regression): {rmse}")

    
    def predict_price(city, bhk, furnishing, property_type):
        # Create a new data frame with the same structure as X
        input_data = pd.DataFrame(columns=X.columns)
        input_data.loc[0] = 0  # Initialize all values to 0
        
        # Set BHK value
        input_data['BHK'] = bhk
        
        # One-hot encode categorical fields based on the given inputs
        if f'City_{city}' in input_data.columns:
            input_data[f'City_{city}'] = 1
        if f'Furnishing_{furnishing}' in input_data.columns:
            input_data[f'Furnishing_{furnishing}'] = 1
        if f'Property_Type_{property_type}' in input_data.columns:
            input_data[f'Property_Type_{property_type}'] = 1

        # Predict the price
        predicted_price = model.predict(input_data)[0]

        if predicted_price < 0:
            predicted_price = abs(predicted_price)
        
        # Format and return the price
        if predicted_price >= 100:
            return f"{predicted_price / 100:.2f} Cr"
        else:
            return f"{predicted_price:.2f} Lac"

    # Example usage
    predicted_price = predict_price(city, bhk, furnishing, property_type)
    print("Predicted price:", predicted_price)
    


    return render(request, 'prediction.html', {'predicted_price': predicted_price})

