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

