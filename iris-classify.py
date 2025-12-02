# Iris Flower Classification - Simple ML Project
# Author: Akanksha

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset
data = load_iris()
X = data.data
y = data.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Prediction & accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", acc)

# 6. Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)

print("Model saved as model.pkl")
