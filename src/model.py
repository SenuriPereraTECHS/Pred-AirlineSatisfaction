import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the split datasets
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Separate features and target
X_train = train_data.drop('satisfaction', axis=1)
y_train = train_data['satisfaction']
X_test = test_data.drop('satisfaction', axis=1)
y_test = test_data['satisfaction']

# 1. Random Forest Classifier
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# 2. Gradient Boosting Classifier
print("Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_preds)

# Printing Results
print("\n--- Model Performance Comparison ---")
print(f"Random Forest Accuracy:     {rf_acc:.4f}")
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_preds))

print("\n--- Gradient Boosting Classification Report ---")
print(classification_report(y_test, gb_preds))

# Confusion Matrix Visualization
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Random Forest Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, gb_preds), annot=True, fmt='d', cmap='Oranges', ax=ax[1])
ax[1].set_title('Gradient Boosting Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('model_comparison_results.png')

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 1. Define the base model
rf = RandomForestClassifier(random_state=42)

# 2. Define the parameter grid to explore
# We focus on parameters that control model complexity (overfitting)
param_grid = {
    'n_estimators': [100, 200],      # Number of trees
    'max_depth': [None, 10, 20],     # Limits how deep trees grow
    'min_samples_split': [2, 5, 10], # Minimum samples required to split a node
    'max_features': ['sqrt', 'log2'] # Number of features considered at each split
}

# 3. Set up GridSearchCV with 5-fold Cross-Validation
# 'cv=5' handles the cross-validation internally
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           scoring='accuracy',
                           verbose=2)

# 4. Fit the model
grid_search.fit(X_train, y_train)

# 5. Extract the best model
best_rf_model = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# 6. Final Evaluation on Test Set
final_preds = best_rf_model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)

print("\n--- Optimized Random Forest Performance ---")
print(f"Final Test Accuracy: {final_acc:.4f}")
print(classification_report(y_test, final_preds))

# 7. Compare with the original model
print(f"Improvement over baseline: {(final_acc - rf_acc):.4f}")