import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/symptoms.csv")
df.fillna(0, inplace=True)

X = df.drop("disease", axis=1)
y_raw = df["disease"]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Train model (on app startup)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def predict_disease(symptom_dict):
    input_df = pd.DataFrame([symptom_dict], columns=X.columns)
    probs = model.predict_proba(input_df)[0]
    best_idx = probs.argmax()
    disease = label_encoder.inverse_transform([best_idx])[0]
    return disease, probs
