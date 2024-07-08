from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load('random_forest_model.pkl')

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form and convert to float
    bedrooms = float(request.form['bedrooms'])
    sqft = float(request.form['sqft'])
    ppsqft = float(request.form['ppsqft'])
    wr = float(request.form['wr'])
    year = float(request.form['year'])
    age = float(request.form['age'])
    primerate = float(request.form['primerate'])
    city = float(request.form['city'])
    type_category = float(request.form['type'])
    walkscore = float(request.form['walkscore'])

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Total Bedrooms': [bedrooms],
        'SqFt Numeric': [sqft],
        'ppsqft': [ppsqft],
        'WR': [wr],
        'Year': [year],
        'Age Numeric': [age],
        'PrimeRate': [primerate],
        'City Category': [city],
        'Type Category': [type_category],
        'Walk Score': [walkscore]
    })

    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the result
    return render_template('index.html', prediction_text='Estimated House Price: ${:,.2f}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
