import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
df = pd.read_csv("data/symptoms.csv")
df.fillna(0, inplace=True)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Debug: print columns
print("CSV Columns:", df.columns.tolist())

# Assume last column is target
target_col = df.columns[-1]

# Extract features
X = df.drop(columns=[target_col], errors='ignore')

# Convert all features to numeric (0/1)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Ensure X is a numpy array of float
X = np.array(X, dtype=float)

# Labels
y_raw = df[target_col]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction function
def predict_disease(symptom_dict):
    """
    Input: symptom_dict -> dictionary with symptom columns as keys and 0/1 as values
    Output: tuple -> (predicted disease string, probabilities array)
    """
    input_df = pd.DataFrame([symptom_dict])

    # Align columns with training features
    input_df = input_df.reindex(columns=df.columns[:-1], fill_value=0)
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    input_array = np.array(input_df, dtype=float)

    probs = model.predict_proba(input_array)[0]
    best_idx = probs.argmax()
    disease = label_encoder.inverse_transform([best_idx])[0]
    return disease, probs
