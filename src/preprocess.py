import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split



# Load the dataset
df = pd.read_csv('Invistico_Airline.csv')
    
#Convert categorical columns to numerical 
'''map the target variable'''
df['satisfaction'] = df['satisfaction'].map({'satisfied' : 1, 'dissatisfied' : 0})

'''encode other categorical variables'''
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Customer Type'] = df['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
df['Type of Travel'] = df['Type of Travel'].map({'Business travel': 1, 'Personal Travel': 0})
df['Class'] = df['Class'].map({'Business': 2, 'Eco Plus': 1, 'Eco': 0})

#check after mapping the variables
print('------------data after mapping categorical variables-------------')
print(df.head())

#check for null values
print('------------checking for null values-------------')
null_counts = df.isna().sum()
print(null_counts)

#handle missing values 
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
print('------------data after handling missing values-------------')
print(df.isna().sum())

# Save this version
df.to_csv('Preprocessed_Airline_Data_V2.csv', index=False)


#After the EDA and Feature Extraction steps, we can now split the dataset into training and testing sets.
def split_dataset(input_file, train_file='train_data.csv', test_file='test_data.csv', test_size=0.2, random_state=42):
    """
    Splits the preprocessed dataset into training and testing sets.
    """
    # Load the final preprocessed dataset
    df = pd.read_csv(input_file)
    
    # Define features (X) and target (y)
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Combine back for easy saving (optional, but good for keeping targets with features)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save the sets
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"Dataset split successfully!")
    print(f"Training set shape: {train_data.shape}")
    print(f"Testing set shape: {test_data.shape}")
    print(f"Target distribution in Train:\n{y_train.value_counts(normalize=True)}")
    print(f"Target distribution in Test:\n{y_test.value_counts(normalize=True)}")
    
    return train_data, test_data

# Execute the split
train_set, test_set = split_dataset('Airline_Data_Final_Features.csv')