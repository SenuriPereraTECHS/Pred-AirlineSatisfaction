import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np


def main():
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

    sns.heatmap(confusion_matrix(y_test, rf_preds),
                annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title('Random Forest Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')

    sns.heatmap(confusion_matrix(y_test, gb_preds),
                annot=True, fmt='d', cmap='Oranges', ax=ax[1])
    ax[1].set_title('Gradient Boosting Confusion Matrix')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('model_comparison_results.png')

    # Hyperparameter Tuning for Random Forest using GridSearchCV

    # 1. Define a constrained parameter grid to reduce memory use
    # Limit tree depth and reduce candidate combinations to avoid excessive memory during training
    param_grid = {
        'n_estimators': [50, 100],                # fewer trees
        # avoid None (unbounded depth)
        'max_depth': [10, 20],
        # larger splits -> smaller trees
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 2],               # prevents tiny leaves
        'max_features': ['sqrt', 'log2']          # features per split
    }

    # 2. Set up GridSearchCV with smaller CV and single-process fitting
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=1),
        param_grid=param_grid,
        cv=3,                 # fewer folds -> smaller training partitions
        n_jobs=1,             # avoid parallel process overhead on Windows
        scoring='accuracy',
        verbose=2,
        error_score=np.nan
    )

    # 3. Fit the model with a memory-safe fallback
    try:
        grid_search.fit(X_train, y_train)
        best_rf_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    except MemoryError:
        print("MemoryError during GridSearchCV. Falling back to RandomizedSearchCV on a subset.")
        from sklearn.model_selection import RandomizedSearchCV

        # If the dataset is large, sample a subset for the randomized search
        subset_size = 60000 if len(X_train) > 60000 else len(X_train)
        X_sub = X_train.sample(subset_size, random_state=42)
        y_sub = y_train.loc[X_sub.index]

        rand_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=1),
            param_distributions=param_grid,
            n_iter=8,
            cv=3,
            n_jobs=1,
            scoring='accuracy',
            verbose=2,
            random_state=42,
            error_score=np.nan
        )

        rand_search.fit(X_sub, y_sub)
        best_rf_model = rand_search.best_estimator_
        print(f"Best Parameters (Randomized): {rand_search.best_params_}")
        print(f"Best CV Score (Randomized): {rand_search.best_score_:.4f}")

    # 5. Final Evaluation on Test Set
    final_preds = best_rf_model.predict(X_test)
    final_acc = accuracy_score(y_test, final_preds)

    print("\n--- Optimized Random Forest Performance ---")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(classification_report(y_test, final_preds))

    # 6. Compare with the original model
    print(f"Improvement over baseline: {(final_acc - rf_acc):.4f}")

    # Save the model
    joblib.dump(best_rf_model, 'best_satisfaction_model.pkl')

    # Save the column(features) names (Crucial for the UI)
    joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

    print("Ready for UI! Model and columns exported.")


if __name__ == "__main__":
    main()
