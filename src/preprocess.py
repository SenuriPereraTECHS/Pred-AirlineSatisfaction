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

