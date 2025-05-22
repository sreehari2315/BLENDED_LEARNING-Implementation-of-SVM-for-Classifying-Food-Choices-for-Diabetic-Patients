## BLENDED LEARNING
## Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients
### DATE:08-05-2025
## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

## Algorithm

1.  **Load Data:** Read `food_items_binary.csv` into a pandas DataFrame.
2.  **Explore (Optional):** Display initial rows and column names.
3.  **Select Features & Target:** Define features (Calories, Fat, Sat. Fat, Sugars, Fiber, Protein) and the binary target ('class'). Separate into `X` and `y`.
4.  **Split Data:** Use `train_test_split` (e.g., 70% train, 30% test, `random_state=42`).
5.  **Scale Features:** Initialize and fit `StandardScaler` on training data, then transform both training and testing sets.
6.  **Tune & Train SVM:**
    * Initialize `SVC()`.
    * Define `param_grid` for 'C', 'kernel' ('linear', 'rbf'), and 'gamma' ('scale', 'auto').
    * Use `GridSearchCV` (with `cv=5`, `scoring='accuracy'`) to find the best hyperparameters by fitting on the training data.
    * Extract the `best_estimator_` and print `best_params_`.
7.  **Evaluate Model:**
    * Predict on the test set using `best_model.predict(X_test)`.
    * Calculate and print `accuracy_score`.
    * Print `classification_report`.
    * Generate and display a `confusion_matrix` using `seaborn.heatmap`.
8.  **Result:** Conclude that the SVM model was successfully implemented and optimized for food classification.

## Program:
```

Program to implement SVM for food classification for diabetic patients.
Developed by: SREE HARI K
RegisterNumber: 212223230212

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset from the URL
data = pd.read_csv('food_items_binary.csv')

# Step 2: Data Exploration
# Display the first few rows and column names for verification
print(data.head())
print(data.columns)

# Step 3: Selecting Features and Target
# Define relevant features and target column
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'  # Assuming 'class' is binary (suitable or not suitable for diabetic patients)

X = data[features]
y = data[target]

# Step 4: Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model Training with Hyperparameter Tuning using GridSearchCV
# Define the SVM model
svm = SVC()

# Set up hyperparameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10, 100],              # Regularization parameter
    'kernel': ['linear', 'rbf'],         # Kernel types
    'gamma': ['scale', 'auto']           # Kernel coefficient for 'rbf'
}

# Initialize GridSearchCV
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Extract the best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Step 7: Model Evaluation
# Predicting on the test set using the best model
y_pred = best_model.predict(X_test)

# Calculate accuracy and print classification metrics
print()
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
### PREVIEW OF THE DATASET:
![image](https://github.com/user-attachments/assets/18c67136-7844-47a2-9dcd-c55f253c94b0)
### EXTRACTION OF BEST MODEL:
![image](https://github.com/user-attachments/assets/eb1d9684-9d1f-4ed1-836e-9abd5ffc5842)

### EVALUATION OF THE MODEL:
![image](https://github.com/user-attachments/assets/742d5079-d66e-4a61-a4da-aba775b4317a)
### CONFUSION MATRIX:
![image](https://github.com/user-attachments/assets/c6efafdc-805e-487b-8208-a6988445676a)


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
