# predict_model.py

import pandas as pd
import joblib
import sqlite3
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Loading the trained model
model = joblib.load('random_forest_model.pkl')  

# Loading the column names used during training 
model_columns = pd.read_csv('model_columns.csv')['column_names'].tolist() 

# Connecting to the database to fetch the data for prediction
conn = sqlite3.connect('C:\\Users\\Sarabjeet Kour\\Database (1).db')  
data = pd.read_sql_query('SELECT * FROM Automobile_data', conn)

print("Data loading successful... initiating prediction")
print()

# Cleaning column names 
data.columns = data.columns.str.strip()

# Handling missing values for categorical columns
data['Ft'] = data['Ft'].fillna(data['Ft'].mode()[0])  
data['Fm'] = data['Fm'].fillna(data['Fm'].mode()[0])  

# Removing rows where the target variable 'Fuel consumption' is missing
data = data[data['Fuel consumption'].notnull()]

# Imputing missing values for numerical columns
numerical_columns = ['m (kg)', 'Mt', 'Ewltp (g/km)', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'Erwltp (g/km)']
imputer = SimpleImputer(strategy='median')  
data[numerical_columns] = imputer.fit_transform(data[numerical_columns]) 

# Dropping unnecessary columns 
columns_to_drop = ['z (Wh/km)', 'Erwltp (g/km)'] 
data = data.drop(columns=columns_to_drop, errors='ignore')

# Using One-hot encoding for categorical columns ('Ft' and 'Fm')
data_encoded = pd.get_dummies(data, columns=['Ft', 'Fm'], drop_first=True) 

# Removing the target columns 'Fuel consumption' and 'Electric range (km)' from data_encoded 
data_encoded = data_encoded.drop(columns=['Fuel consumption', 'Electric range (km)'], errors='ignore')  

# Adding any missing columns with a default value of 0 
for column in model_columns:
    if column not in data_encoded.columns:
        data_encoded[column] = 0  
        
# Re-ordering columns to match the original column order used during training 
data_encoded = data_encoded[model_columns]

# Now the data is ready for prediction
X_prod = data_encoded  

# Making predictions
y_pred = model.predict(X_prod)  

# Displaying the first 10 predictions
print(f"Predictions:\n{y_pred[:10]}") 


