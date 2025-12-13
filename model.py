import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/symptoms.csv")
df.fillna(0, inplace=True)

# Automatically detect target column
if "disease" in df.columns:
    target_col = "disease"
elif "prognosis" in df.columns:
    target_col = "prognosis"
else:
    raise KeyError(f"Target column not found. Available columns: {list(df.columns)}")

# Features and labels
X = df.drop(target_col, axis=1)
y_raw = df[target_col]

# Encode disease labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Train Random Forest model at runtime
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction function
def predict_disease(symptom_dict):
    """
    Input: symptom_dict -> dictionary with column names as keys and 0/1 as values
    Output: tuple -> (predicted disease string, probabilities array)
    """
    input_df = pd.DataFrame([symptom_dict], columns=X.columns)
    probs = model.predict_proba(input_df)[0]
    best_idx = probs.argmax()
    disease = label_encoder.inverse_transform([best_idx])[0]
    return disease, probs
