import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def feature_extraction_and_correlation(df, output_heatmap = 'correlationHeatmap.png'):
    """
    Analyzes feature correlations, visualizes them, and performs basic feature extraction.
    """
    # 1. Correlation Analysis
    corr_matrix = df.corr()
    
    # 2. Plotting Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Airline Features')
    plt.tight_layout()
    plt.savefig(output_heatmap)
    
    # 3. Feature Extraction (Engineering)
    # Since Departure and Arrival delays are highly correlated (0.96), 
    # we can combine them into one 'Total Delay' feature.
    print("Extracting new feature: 'Total Delay'...")
    df['Total Delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    
    # Drop the redundant columns to reduce dimensionality and multicollinearity
    df_extracted = df.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1)
    
    # 4. Identifying top correlated features with Satisfaction
    satisfaction_corr = corr_matrix['satisfaction'].sort_values(ascending=False)
    print("\n--- Features most correlated with Satisfaction ---")
    print(satisfaction_corr)
    
    return df_extracted, satisfaction_corr

# Load the data from the previous step
df_preprocessed = pd.read_csv('Preprocessed_Airline_Data_V2.csv')

# Execute extraction and correlation analysis
df_final, satisfaction_correlations = feature_extraction_and_correlation(df_preprocessed)

# Save the final dataset
df_final.to_csv('Airline_Data_Final_Features.csv', index=False)
print("\nFinal feature-extracted data saved to: Airline_Data_Final_Features.csv")