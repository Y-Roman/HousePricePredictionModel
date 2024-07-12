# House Price Prediction Model

This repository contains a machine learning model for predicting house prices in the Halton Region. The model is deployed using a Flask web application, allowing users to input house features and receive an estimated price.

## Features

- Trains multiple machine learning models and selects the best performing one (Random Forest).
- A web tool that allows users to input features and estimate house prices.
- Utilizes Flask for the web interface.
- Features for the home include Home Type, Number of Beds, Number of Washrooms, Square Feet, Prime Rate and walk score 

## Installation Instructions

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Y-Roman/HousePricePredictionModel.git
cd HousePricePredictionModel

pip install Flask numpy pandas joblib scikit-learn
```

### 2. Run the Flask App

```bash
python ml.py
```

![Example](https://github.com/Y-Roman/HousePricePredictionModel/tree/master/images/sampleUI.jpg?raw=true)
