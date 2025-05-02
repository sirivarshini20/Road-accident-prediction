# Importing all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Create a directory for saving models if it doesn't exist
model_dir = 'saved_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load the dataset (assuming you have LabelEncoded.csv in your working directory)
df = pd.read_csv("LabelEncoded.csv")
df = df.drop(columns=['Unnamed: 0'])

# Separate features and target
y = df['Severity']
X = df.drop('Severity', axis=1)

# Feature scaling and train-test split
sc = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Save the scaler for future use
joblib.dump(sc, f'{model_dir}/standard_scaler.pkl')

# Function to evaluate model performance
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, f'{model_dir}/{model_name.replace(" ", "_").lower()}.pkl')
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"=== {model_name} ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved as: {model_dir}/{model_name.replace(' ', '_').lower()}.pkl")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {test_accuracy:.4f}')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/{model_name.replace(" ", "_").lower()}_cm.png')
    plt.show()
    
    return train_accuracy, test_accuracy, y_pred

# Create a dictionary to store models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Dictionary to store results
results = {}

# Evaluate each model
for name, model in models.items():
    print("\n" + "="*50)
    train_acc, test_acc, predictions = evaluate_model(model, name, X_train_scaled, X_test_scaled, y_train, y_test)
    results[name] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'predictions': predictions
    }

# Save the model comparison results
comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Training Accuracy': [results[model]['train_accuracy'] for model in models],
    'Test Accuracy': [results[model]['test_accuracy'] for model in models]
})
comparison_df.to_csv(f'{model_dir}/model_comparison_results.csv', index=False)
print(f"\nModel comparison results saved to {model_dir}/model_comparison_results.csv")

# Compare model performances with a bar chart
train_accuracies = [results[model]['train_accuracy'] for model in models]
test_accuracies = [results[model]['test_accuracy'] for model in models]

plt.figure(figsize=(14, 8))
x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, train_accuracies, width, label='Training Accuracy')
plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(x, models.keys(), rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{model_dir}/model_comparison.png')
plt.show()

# Let's find the best performing model on the test set
best_model_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
best_test_acc = results[best_model_name]['test_accuracy']
print(f"\nBest performing model: {best_model_name} with test accuracy: {best_test_acc:.4f}")

# Feature importance analysis for tree-based models
for model_name in ["Random Forest", "Gradient Boosting"]:
    if model_name in results:
        # Get feature importances
        model = models[model_name]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances from {model_name}')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'{model_dir}/{model_name.replace(" ", "_").lower()}_feature_importance.png')
        plt.show()
        
        # Save feature importances to CSV
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns[indices],
            'Importance': importances[indices]
        })
        feature_importance_df.to_csv(f'{model_dir}/{model_name.replace(" ", "_").lower()}_feature_importance.csv', index=False)
        print(f"Feature importances for {model_name} saved to {model_dir}/{model_name.replace(' ', '_').lower()}_feature_importance.csv")

print(f"\nAll models, results, and visualizations have been saved to the '{model_dir}' directory.")








