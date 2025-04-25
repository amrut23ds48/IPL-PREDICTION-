import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# Load the dataset
df = pd.read_csv("merge_csv.csv")
df = df[df['venue'] != 'OUT']  # Remove rows where venue is "OUT"

# Drop unnecessary columns
columns_to_drop = ['mvp']
df = df.drop(columns=columns_to_drop)

# Split features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert categorical features to numerical using One-Hot Encoding
X = pd.get_dummies(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize RandomForestClassifier with tuned hyperparameters
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# **1. Feature Importance Visualization**
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance in RandomForest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# **2. Permutation Importance (More Reliable Feature Impact)**
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10, 5))
plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=np.array(features)[sorted_idx])
plt.title("Permutation Importance")
plt.xlabel("Impact on Model Performance")
plt.show()

