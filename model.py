import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/symptoms.csv")
df.fillna(0, inplace=True)

# Strip spaces and lowercase column names
df.columns = df.columns.str.strip().str.lower()

# Debug: print columns
print("CSV Columns:", df.columns.tolist())

# Use last column as target
target_col = df.columns[-1]

# Features and labels
X = df.drop(columns=[target_col], errors="ignore")
y_raw = df[target_col] if target_col in df.columns else None

if y_raw is None:
    raise KeyError(f"Target column '{target_col}' not found in CSV. Columns: {df.columns.tolist()}")

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

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
    input_df = pd.DataFrame([symptom_dict])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Ensure numeric
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    probs = model.predict_proba(input_df)[0]
    best_idx = probs.argmax()
    disease = label_encoder.inverse_transform([best_idx])[0]
    return disease, probs
