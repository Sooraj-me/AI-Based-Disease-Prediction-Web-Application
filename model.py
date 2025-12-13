import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/symptoms.csv")
df.fillna(0, inplace=True)

# -----------------------------
# Set the target column here:
# Replace 'prognosis' with the actual target column from your CSV
# -----------------------------
target_col = "prognosis"  

# Features and labels
X = df.drop(target_col, axis=1)
y_raw = df[target_col]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Train Random Forest at runtime
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction function
def predict_disease(symptom_dict):
    """
    Input: symptom_dict -> dictionary with symptom columns as keys and 0/1 as values
    Output: tuple -> (predicted disease string, probabilities array)
    """
    input_df = pd.DataFrame([symptom_dict], columns=X.columns)
    probs = model.predict_proba(input_df)[0]
    best_idx = probs.argmax()
    disease = label_encoder.inverse_transform([best_idx])[0]
    return disease, probs
