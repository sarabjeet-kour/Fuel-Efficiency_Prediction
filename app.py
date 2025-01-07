# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import sqlite3

# Initializing Flask app
app = Flask(__name__)

# Loading the trained model
model = joblib.load('random_forest_model.pkl')  

# Loading the column names used during training
model_columns = pd.read_csv('model_columns.csv')['column_names'].tolist()  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting input data from the request
        data = request.get_json()

        # Checking if the input data is provided
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Ensuring the data is in the correct format 
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, list): 
                    data[key] = [value]

        # Checking if all columns in the input have the same length
        lengths = [len(value) for value in data.values()]
        if len(set(lengths)) != 1:
            return jsonify({"error": "All input columns must have the same length"}), 400

        # Converting the input data to a DataFrame
        input_data = pd.DataFrame(data)

        # Cleaning column names (remove leading/trailing spaces)
        input_data.columns = input_data.columns.str.strip()

        # Handling missing values for categorical columns
        input_data['Ft'] = input_data['Ft'].fillna(input_data['Ft'].mode()[0])  
        input_data['Fm'] = input_data['Fm'].fillna(input_data['Fm'].mode()[0]) 

        # Handling missing values for numerical columns using median imputation
        numerical_columns = ['m (kg)', 'Mt', 'Ewltp (g/km)', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'Erwltp (g/km)']
        imputer = SimpleImputer(strategy='median')
        existing_numerical_columns = [col for col in numerical_columns if col in input_data.columns]
        input_data[existing_numerical_columns] = imputer.fit_transform(input_data[existing_numerical_columns])

        # Imputing missing values for 'Electric range (km)' based on fuel type
        input_data['Electric range (km)'] = input_data.apply(lambda row: 0 if row['Ft'] != 'electric' else row['Electric range (km)'], axis=1)
        input_data['Electric range (km)'] = input_data['Electric range (km)'].fillna(input_data['Electric range (km)'].median())

        # One-hot encode categorical columns ('Ft' and 'Fm')
        input_data_encoded = pd.get_dummies(input_data, columns=['Ft', 'Fm'], drop_first=True)

        # Dropping target columns ('Fuel consumption' and 'Electric range (km)') for prediction
        input_data_encoded = input_data_encoded.drop(columns=['Fuel consumption', 'Electric range (km)'], errors='ignore')

        # Adding any missing columns with a default value of 0
        for column in model_columns:
            if column not in input_data_encoded.columns:
                input_data_encoded[column] = 0  

        # Re-ordering columns to match the original order used during training
        input_data_encoded = input_data_encoded[model_columns]

        # Making predictions using the model
        predictions = model.predict(input_data_encoded)

        # Returning predictions as a JSON response
        return jsonify(predictions.tolist())

    except Exception as e:
        # Logging the error and returning a response indicating failure
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
