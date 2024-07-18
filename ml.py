from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the Models for different datasets
halton_model = load('Models/Halton_RFM.pkl')
peel_model = load('Models/Peel_RFM.pkl')

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/halton_index')
def halton_index():
    return render_template('halton_index.html')

@app.route('/peel_index')
def peel_index():
    return render_template('peel_index.html')
    
# Define the prediction route for Halton Region
@app.route('/predict_halton', methods=['POST'])
def predict_halton():
    # Get the input features from the form and convert to float
    bedrooms = float(request.form['bedrooms'])
    sqft = float(request.form['sqft'])
    wr = float(request.form['wr'])
    age = float(request.form['age'])
    primerate = float(request.form['primerate'])
    city = float(request.form['city'])
    type_category = float(request.form['type'])
    walkscore = float(request.form['walkscore'])
    
    # Map walkscore_category to the average value of the range
    walkscore_map = {
        0: (90 + 100) / 2,  # Walker’s Paradise
        1: (70 + 89) / 2,   # Very Walkable
        2: (50 + 69) / 2,   # Somewhat Walkable
        3: (25 + 49) / 2,   # Car-Dependent (Most errands require a car)
        4: (0 + 24) / 2     # Car-Dependent (Almost all errands require a car)
    }
    walkscore = walkscore_map[walkscore]
    
    # Create a DataFrame from the input data
    input_data_halton = pd.DataFrame({
        'Total Bedrooms': [bedrooms],
        'SqFt Numeric': [sqft],
        'WR': [wr],
        'Age Numeric': [age],
        'PrimeRate': [primerate],
        'City Category': [city],
        'Type Category': [type_category],
        'Walk Score': [walkscore]
    })

    # Make prediction using the Halton Region model
    prediction = halton_model.predict(input_data_halton)

    # Return the result for Halton Region
    return render_template('halton_index.html', prediction_text='Estimated House Price for Halton Region: ${:,.2f}'.format(prediction[0]))

# Define the prediction route for Peel Region
@app.route('/predict_peel', methods=['POST'])
def predict_peel():
    # Get the input features from the form and convert to float
    bedrooms = float(request.form['bedrooms'])
    sqft = float(request.form['sqft'])
    wr = float(request.form['wr'])
    age = float(request.form['age'])
    city = float(request.form['city'])
    type_category = float(request.form['type'])
    
    # Create a DataFrame from the input data (excluding Prime Rate and Walk Score)
    input_data_peel = pd.DataFrame({
        'Bedrooms Total': [bedrooms],
        'SqFt Numeric': [sqft],
        'WR': [wr],
        'Age Numeric': [age],
        'City Category': [city],
        'Type Category': [type_category]
    })

    # Make prediction using the Peel Region model
    prediction = peel_model.predict(input_data_peel)

    # Return the result for Peel Region
    return render_template('peel_index.html', prediction_text='Estimated House Price for Peel Region: ${:,.2f}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
